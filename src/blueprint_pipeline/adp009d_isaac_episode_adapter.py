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
* **Gripper width.**  The scorer reads a finger separation in metres, which is
  the same quantity the gripper-convention probe measures, so it is read from
  the same two finger bodies rather than inferred from a joint angle.

Isaac imports happen inside methods, never at module import, so this file stays
importable -- and testable -- off-GPU.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .adp009d_droid_observation import (
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
)

ADAPTER_SCHEMA_VERSION = "adp009d_isaac_episode_adapter.v1"

# Isaac camera name -> the DROID view it serves.
CAMERA_VIEW_BINDING = {
    "external_camera": DROID_EXTERIOR_VIEW_1,
    "wrist_camera": DROID_WRIST_VIEW,
}
# The two bodies whose separation is the gripper width, matching the
# convention probe so both read the same physical quantity.
FINGER_BODIES = ("left_inner_finger", "right_inner_finger")
ARM_JOINT_COUNT = 7


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
    ) -> None:
        self._env = env
        self._robot = robot
        self._can = approved_can
        self._action_dim = int(action_dim)
        self._reset_seed = int(reset_seed)
        self._to_torch = to_torch

        body_names = list(robot.data.body_names)
        missing = [name for name in FINGER_BODIES if name not in body_names]
        if missing:
            raise IsaacEpisodeAdapterError(
                [f"isaac_episode_finger_body_missing:{','.join(missing)}"]
            )
        self._finger_indices = [body_names.index(name) for name in FINGER_BODIES]

    # -- EpisodeEnvironment -------------------------------------------------

    def reset(self) -> None:
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
                raise IsaacEpisodeAdapterError(
                    [f"isaac_episode_camera_rgb_missing:{camera_name}"]
                )
            frame = _as_array(self._to_torch(output["rgb"]))[0]
            inputs[view] = rgb_from_camera_output(frame)
        joints = self._to_torch(self._robot.data.joint_pos)[0, :ARM_JOINT_COUNT]
        inputs["joint_position"] = [float(v) for v in joints]
        inputs["gripper_position"] = self._gripper_width()
        return inputs

    def step(self, isaac_action: Sequence[float]) -> None:
        import torch

        values = [float(v) for v in isaac_action]
        if len(values) != self._action_dim:
            raise IsaacEpisodeAdapterError(
                [f"isaac_episode_action_dim_mismatch:{len(values)}!={self._action_dim}"]
            )
        tensor = torch.tensor(
            [values], device=self._env.unwrapped.device, dtype=torch.float32
        )
        self._env.step(tensor)

    def read_object_sample(self) -> dict[str, Any]:
        pose = self._to_torch(self._can.data.root_pose_w)[0]
        sample: dict[str, Any] = {
            "can_pose_world": [float(v) for v in pose[:7]],
            "gripper_width_m": self._gripper_width(),
        }
        left, right = self._finger_positions()
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

    def _gripper_width(self) -> float:
        # A Euclidean distance needs no tensor library, and keeping it in plain
        # arithmetic is what lets this adapter be tested without a GPU.
        left, right = self._finger_positions()
        return math.dist(left, right)


def describe_adapter() -> dict[str, Any]:
    """Report the bindings this adapter applied, for the run receipt."""

    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "camera_view_binding": dict(CAMERA_VIEW_BINDING),
        "finger_bodies": list(FINGER_BODIES),
        "gripper_width_source": "finger_body_separation",
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
    if bindings.get("camera_alpha_dropped_at_boundary") is not True:
        errors.append("isaac_episode_adapter_alpha_not_dropped")
    return sorted(set(errors))


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "CAMERA_VIEW_BINDING",
    "FINGER_BODIES",
    "IsaacEpisodeAdapter",
    "IsaacEpisodeAdapterError",
    "describe_adapter",
    "rgb_from_camera_output",
    "validate_adapter_bindings",
]
