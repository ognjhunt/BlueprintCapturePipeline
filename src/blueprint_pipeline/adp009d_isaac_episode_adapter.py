"""Adapt the live Isaac environment to the episode loop's injected seam.

``adp009d_policy_episode`` deliberately never imports a simulator, so something
has to translate between it and Isaac.  That is this module, and it is kept
thin on purpose: every non-trivial decision -- observation format, action
semantics, scoring -- already lives in a tested module, and duplicating any of
it here would create a second place for those contracts to drift.

Two conversions are the substance of it, and both are measured facts rather
than conventions:

* **Camera frames.**  Isaac renders RGBA at 1280x720; the DROID views want
  uint8 RGB, which the observation adapter then resizes per candidate.  Alpha
  is dropped here rather than downstream because a constant alpha channel
  silently dominated a colour-signal check once already.
* **Gripper width.**  The scorer reads physical jaw opening in metres.  The
  Robotiq link origins are not jaw tips and can dynamically travel beyond the
  85 mm nameplate stroke, so their separation is affine-calibrated against the
  same run's measured closed/open convention probe.  The raw separation and a
  clamp flag remain in every object sample for audit.

Isaac imports happen inside methods, never at module import, so this file stays
importable -- and testable -- off-GPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:  # flat provider-bundle layout
    from adp009d_droid_observation import (
        DROID_EXTERIOR_VIEW_1,
        DROID_WRIST_VIEW,
    )
except ModuleNotFoundError:  # repository package
    from .adp009d_droid_observation import (
        DROID_EXTERIOR_VIEW_1,
        DROID_WRIST_VIEW,
    )

ADAPTER_SCHEMA_VERSION = "adp009d_isaac_episode_adapter.v2"

# Isaac camera name -> the DROID view it serves.
CAMERA_VIEW_BINDING = {
    "external_camera": DROID_EXTERIOR_VIEW_1,
    "wrist_camera": DROID_WRIST_VIEW,
}
# The two bodies whose separation is the gripper width, matching the
# convention probe so both read the same physical quantity.
FINGER_BODIES = ("left_inner_finger", "right_inner_finger")
# Ordered to match the already-measured approach controller.  ``base_link`` is
# the Robotiq tool body that carries the wrist camera in the live Arena asset.
END_EFFECTOR_BODY_CANDIDATES = ("panda_hand", "base_link", "panda_link7")
ARM_JOINT_COUNT = 7
# Frozen by the Robotiq 2F-85 task embodiment and pinned independently by the
# deterministic scorer.  A parity test keeps the flat-bundle duplicate honest.
GRIPPER_PHYSICAL_FULL_OPENING_M = 0.085
GRIPPER_WIDTH_SOURCE = "probe_calibrated_finger_body_separation"


class IsaacEpisodeAdapterError(RuntimeError):
    """Fail-closed adapter errors, raised before anything reaches the loop."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def _as_array(value: Any) -> Any:
    """Whatever the simulator handed back, as a numpy array.

    Live Isaac returns a torch tensor; tests and replays hand back arrays.  This
    is the adapter boundary, so it accepts both rather than assuming one and
    failing obscurely on the other.
    """

    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def rgb_from_camera_output(frame: Any) -> Any:
    """Isaac RGBA (or RGB) to contiguous uint8 RGB.

    Alpha is dropped here, at the boundary, because a constant alpha channel
    has already once dominated a statistic computed over all four channels and
    made a black render look like it carried signal.
    """

    import numpy as np

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise IsaacEpisodeAdapterError(
            [f"isaac_episode_camera_frame_shape_invalid:{tuple(array.shape)}"]
        )
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def rotation_row_major_from_quaternion_wxyz(quaternion: Sequence[float]) -> list[float]:
    """Convert Isaac Lab's normalized ``(w, x, y, z)`` body quaternion."""

    values = [float(value) for value in quaternion]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise IsaacEpisodeAdapterError(["isaac_episode_end_effector_quaternion_invalid"])
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise IsaacEpisodeAdapterError(["isaac_episode_end_effector_quaternion_invalid"])
    w, x, y, z = (value / norm for value in values)
    return [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]


class IsaacEpisodeAdapter:
    """The live-simulator implementation of ``EpisodeEnvironment``."""

    def __init__(
        self,
        *,
        env: Any,
        robot: Any,
        approved_can: Any,
        action_dim: int,
        reset_seed: int,
        to_torch: Any,
        gripper_closed_width_m: float,
        gripper_open_width_m: float,
        reset_callback: Callable[[], None] | None = None,
    ) -> None:
        self._env = env
        self._robot = robot
        self._can = approved_can
        self._action_dim = int(action_dim)
        self._reset_seed = int(reset_seed)
        self._to_torch = to_torch
        self._gripper_closed_width_m = float(gripper_closed_width_m)
        self._gripper_open_width_m = float(gripper_open_width_m)
        self._reset_callback = reset_callback
        if (
            not math.isfinite(self._gripper_closed_width_m)
            or not math.isfinite(self._gripper_open_width_m)
            or self._gripper_open_width_m - self._gripper_closed_width_m <= 1e-6
        ):
            raise IsaacEpisodeAdapterError(
                ["isaac_episode_gripper_width_calibration_invalid"]
            )

        body_names = list(robot.data.body_names)
        missing = [name for name in FINGER_BODIES if name not in body_names]
        if missing:
            raise IsaacEpisodeAdapterError(
                [f"isaac_episode_finger_body_missing:{','.join(missing)}"]
            )
        self._finger_indices = [body_names.index(name) for name in FINGER_BODIES]
        end_effector_name = next(
            (name for name in END_EFFECTOR_BODY_CANDIDATES if name in body_names),
            None,
        )
        if end_effector_name is None:
            raise IsaacEpisodeAdapterError(["isaac_episode_end_effector_body_missing"])
        self._end_effector_name = end_effector_name
        self._end_effector_index = body_names.index(end_effector_name)

    # -- EpisodeEnvironment -------------------------------------------------

    def reset(self) -> None:
        if self._reset_callback is not None:
            self._reset_callback()
        else:
            self._env.reset(seed=self._reset_seed)

    def joint_limits(self) -> list[list[float]]:
        limits = self._to_torch(self._robot.data.joint_limits)[0, :ARM_JOINT_COUNT]
        return [[float(row[0]), float(row[1])] for row in limits]

    def read_policy_inputs(self) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for camera_name, view in CAMERA_VIEW_BINDING.items():
            camera = self._env.unwrapped.scene[camera_name]
            output = camera.data.output
            if "rgb" not in output:
                raise IsaacEpisodeAdapterError([f"isaac_episode_camera_rgb_missing:{camera_name}"])
            frame = _as_array(self._to_torch(output["rgb"]))[0]
            inputs[view] = rgb_from_camera_output(frame)
        inputs["joint_position"] = self.read_arm_joint_positions()
        inputs["gripper_position"] = self._droid_gripper_position()
        inputs["eef_9d"] = self._eef_9d()
        return inputs

    def read_arm_joint_positions(self) -> list[float]:
        joints = self._to_torch(self._robot.data.joint_pos)[0, :ARM_JOINT_COUNT]
        return [float(value) for value in joints]

    def step(self, isaac_action: Sequence[float]) -> None:
        values = [float(v) for v in isaac_action]
        if len(values) != self._action_dim:
            raise IsaacEpisodeAdapterError(
                [f"isaac_episode_action_dim_mismatch:{len(values)}!={self._action_dim}"]
            )

        # Validate before importing the GPU runtime.  Besides producing the
        # precise contract error first, this keeps malformed-action checks
        # hermetic on hosts that deliberately do not install torch.
        import torch

        tensor = torch.tensor([values], device=self._env.unwrapped.device, dtype=torch.float32)
        self._env.step(tensor)

    def read_object_sample(self) -> dict[str, Any]:
        pose = self._to_torch(self._can.data.root_pose_w)[0]
        left, right = self._finger_positions()
        raw_separation = math.dist(left, right)
        width, unclamped_open_fraction, calibration_clamped = (
            self._calibrated_gripper_width(raw_separation)
        )
        sample: dict[str, Any] = {
            "can_pose_world": [float(v) for v in pose[:7]],
            "gripper_width_m": width,
            "gripper_body_separation_m": raw_separation,
            "gripper_width_open_fraction_unclamped": unclamped_open_fraction,
            "gripper_width_calibration_clamped": calibration_clamped,
        }
        sample["grasp_frame_position_world_m"] = [
            (left[axis] + right[axis]) / 2.0 for axis in range(3)
        ]
        return sample

    # -- internals ----------------------------------------------------------

    def _finger_positions(self) -> tuple[list[float], list[float]]:
        poses = self._to_torch(self._robot.data.body_pose_w)[0]
        return (
            [float(poses[self._finger_indices[0]][axis]) for axis in range(3)],
            [float(poses[self._finger_indices[1]][axis]) for axis in range(3)],
        )

    def _raw_gripper_body_separation(self) -> float:
        # A Euclidean distance needs no tensor library, and keeping it in plain
        # arithmetic is what lets this adapter be tested without a GPU.
        left, right = self._finger_positions()
        return math.dist(left, right)

    def _calibrated_gripper_width(
        self, raw_separation_m: float
    ) -> tuple[float, float, bool]:
        """Map linkage separation onto the gripper's physical jaw stroke.

        The convention probe supplies the run-local endpoints.  Values outside
        them are linkage/dynamics overtravel rather than a jaw aperture wider
        than the 2F-85 can physically achieve, so the calibrated aperture is
        bounded while the unbounded fraction is retained for diagnosis.
        """

        span = self._gripper_open_width_m - self._gripper_closed_width_m
        unbounded = (float(raw_separation_m) - self._gripper_closed_width_m) / span
        bounded = min(1.0, max(0.0, unbounded))
        return (
            GRIPPER_PHYSICAL_FULL_OPENING_M * bounded,
            unbounded,
            not math.isclose(unbounded, bounded, rel_tol=0.0, abs_tol=1e-12),
        )

    def _droid_gripper_position(self) -> float:
        """DROID state convention: zero=open and one=closed."""

        _, open_fraction, _ = self._calibrated_gripper_width(
            self._raw_gripper_body_separation()
        )
        return min(1.0, max(0.0, 1.0 - open_fraction))

    def _eef_9d(self) -> Any:
        pose = self._to_torch(self._robot.data.body_pose_w)[
            0, self._end_effector_index
        ]
        values = [float(value) for value in pose[:7]]
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise IsaacEpisodeAdapterError(["isaac_episode_end_effector_pose_invalid"])
        try:  # flat provider bundle
            from groot_n17_droid_policy_runtime import droid_eef_9d
        except ModuleNotFoundError:  # repository package
            from .groot_n17_droid_policy_runtime import droid_eef_9d
        return droid_eef_9d(
            position_m=values[:3],
            rotation_row_major=rotation_row_major_from_quaternion_wxyz(values[3:7]),
        )


def describe_adapter() -> dict[str, Any]:
    """Report the bindings this adapter applied, for the run receipt."""

    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "camera_view_binding": dict(CAMERA_VIEW_BINDING),
        "finger_bodies": list(FINGER_BODIES),
        "end_effector_body_candidates": list(END_EFFECTOR_BODY_CANDIDATES),
        "gripper_width_source": GRIPPER_WIDTH_SOURCE,
        "gripper_physical_full_opening_m": GRIPPER_PHYSICAL_FULL_OPENING_M,
        "raw_gripper_body_separation_retained": True,
        "gripper_width_calibration_clamp_retained": True,
        "camera_alpha_dropped_at_boundary": True,
        "arm_joint_count": ARM_JOINT_COUNT,
    }


def validate_adapter_bindings(bindings: Mapping[str, Any]) -> list[str]:
    """Check a retained adapter description still matches this contract."""

    errors: list[str] = []
    if bindings.get("schema_version") != ADAPTER_SCHEMA_VERSION:
        errors.append("isaac_episode_adapter_schema_version_unexpected")
    if dict(bindings.get("camera_view_binding") or {}) != dict(CAMERA_VIEW_BINDING):
        errors.append("isaac_episode_adapter_camera_binding_drifted")
    if list(bindings.get("finger_bodies") or []) != list(FINGER_BODIES):
        errors.append("isaac_episode_adapter_finger_bodies_drifted")
    if list(bindings.get("end_effector_body_candidates") or []) != list(
        END_EFFECTOR_BODY_CANDIDATES
    ):
        errors.append("isaac_episode_adapter_end_effector_binding_drifted")
    if bindings.get("gripper_width_source") != GRIPPER_WIDTH_SOURCE:
        errors.append("isaac_episode_adapter_gripper_width_source_drifted")
    if (
        bindings.get("gripper_physical_full_opening_m")
        != GRIPPER_PHYSICAL_FULL_OPENING_M
    ):
        errors.append("isaac_episode_adapter_gripper_stroke_drifted")
    if bindings.get("raw_gripper_body_separation_retained") is not True:
        errors.append("isaac_episode_adapter_raw_gripper_measurement_not_retained")
    if bindings.get("gripper_width_calibration_clamp_retained") is not True:
        errors.append("isaac_episode_adapter_gripper_clamp_not_retained")
    if bindings.get("camera_alpha_dropped_at_boundary") is not True:
        errors.append("isaac_episode_adapter_alpha_not_dropped")
    return sorted(set(errors))


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "CAMERA_VIEW_BINDING",
    "END_EFFECTOR_BODY_CANDIDATES",
    "FINGER_BODIES",
    "GRIPPER_PHYSICAL_FULL_OPENING_M",
    "GRIPPER_WIDTH_SOURCE",
    "IsaacEpisodeAdapter",
    "IsaacEpisodeAdapterError",
    "describe_adapter",
    "rgb_from_camera_output",
    "rotation_row_major_from_quaternion_wxyz",
    "validate_adapter_bindings",
]
