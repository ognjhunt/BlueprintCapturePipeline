"""Absolute-joint DROID to OSCAR skeleton-conditioning contracts.

This module constructs the camera-aligned intended-motion representation used
by OSCAR.  It does not predict world pixels, simulate contact, or guess camera
geometry.  Camera calibration and Franka forward kinematics are explicit,
replaceable inputs so the same contract applies to future experiments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .droid_oscar_closed_loop_adapter import WAM_SOURCE_VIEW_PATHS
from .droid_policy_bridge import DROID_ROBOARENA_CONCAT_VIEWS
from .policy_ranking_thesis import canonical_sha256, file_sha256


POLICY_ACTION_SHAPE = (16, 8)
OSCAR_NUM_FRAMES = 81
OSCAR_FPS = 15.0
PANDA_JOINT_LIMITS_RAD = np.asarray(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float64,
)


def _finite_array(value: Any, shape: tuple[int, ...], reason: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(reason)
    return array


def _safe_file(value: Any, reason: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(reason)
    return path


def build_absolute_joint_skeleton_trajectory(
    policy_action: Any,
    *,
    current_joint_position: Sequence[float],
    current_gripper_position: Sequence[float],
    executed_prefix_steps: int,
) -> dict[str, Any]:
    """Expand one 16x8 absolute Policy-DROID chunk to OSCAR's 81 frames.

    Frame zero is the current commanded state, frames 1..16 are the policy's
    absolute targets, and frames 17..80 hold the final target.  Out-of-range
    commands fail closed; this evidence layer never silently clamps policy
    output and then attributes the altered trajectory to the policy.
    """

    action = _finite_array(policy_action, POLICY_ACTION_SHAPE, "invalid_absolute_action_chunk")
    current_joints = _finite_array(
        current_joint_position, (7,), "invalid_current_joint_position"
    )
    current_gripper = _finite_array(
        current_gripper_position, (1,), "invalid_current_gripper_position"
    )
    if isinstance(executed_prefix_steps, bool) or not isinstance(executed_prefix_steps, int):
        raise ValueError("executed_prefix_steps_must_be_integer")
    if not 1 <= executed_prefix_steps <= POLICY_ACTION_SHAPE[0]:
        raise ValueError("executed_prefix_steps_out_of_range")
    all_joints = np.vstack((current_joints[None, :], action[:, :7]))
    if np.any(all_joints < PANDA_JOINT_LIMITS_RAD[:, 0]) or np.any(
        all_joints > PANDA_JOINT_LIMITS_RAD[:, 1]
    ):
        raise ValueError("absolute_joint_target_out_of_range")
    all_gripper = np.concatenate((current_gripper, action[:, 7]))
    if np.any(all_gripper < 0.0) or np.any(all_gripper > 1.0):
        raise ValueError("absolute_gripper_target_out_of_range")

    hold_count = OSCAR_NUM_FRAMES - len(all_joints)
    joint_frames = np.vstack((all_joints, np.repeat(all_joints[-1][None, :], hold_count, axis=0)))
    gripper_frames = np.concatenate(
        (all_gripper, np.repeat(all_gripper[-1], hold_count))
    )
    material = {
        "schema_version": "droid_absolute_joint_skeleton_trajectory.v1",
        "policy_action_shape": list(POLICY_ACTION_SHAPE),
        "joint_angles_rad": joint_frames.tolist(),
        "gripper_openness": gripper_frames.tolist(),
        "executed_prefix_steps": executed_prefix_steps,
        "hold_final_target_after_action_chunk": True,
        "joint_targets_clamped": False,
    }
    return {
        **material,
        "trajectory_sha256": canonical_sha256(material),
        "joint_angles_array": joint_frames,
        "gripper_openness_array": gripper_frames,
        "next_joint_position": action[executed_prefix_steps - 1, :7].copy(),
        "next_gripper_position": action[executed_prefix_steps - 1, 7:8].copy(),
    }


def _validate_se3(poses: np.ndarray, reason: str) -> None:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(reason)
    rotations = poses[:, :3, :3]
    identity = np.eye(3)[None, :, :]
    if not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, identity, atol=1e-5):
        raise ValueError(reason)
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5):
        raise ValueError(reason)
    if not np.allclose(poses[:, 3, :], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError(reason)


def encode_fk_reliability_actions_10d(
    joint_states: Any,
    gripper_states: Any,
    *,
    forward_kinematics: Callable[[np.ndarray], Any],
) -> np.ndarray:
    """Encode verified FK motion as 16 relative 10-D reliability actions.

    Translation is the base-frame position delta. Rotation is the first two
    columns of the relative SO(3) rotation, yielding identity rot6d
    ``[1,0,0,0,1,0]`` for no rotation. This trace measures command presence
    and timing only; it is not claimed as Cosmos's native DROID normalization.
    """

    joints = _finite_array(joint_states, (17, 7), "invalid_fk_joint_states")
    gripper = _finite_array(gripper_states, (17,), "invalid_fk_gripper_states")
    poses = np.asarray([forward_kinematics(row) for row in joints], dtype=np.float64)
    _validate_se3(poses, "invalid_forward_kinematics_pose")
    actions = np.zeros((16, 10), dtype=np.float64)
    for index in range(16):
        prior, current = poses[index], poses[index + 1]
        relative_rotation = prior[:3, :3].T @ current[:3, :3]
        actions[index, :3] = current[:3, 3] - prior[:3, 3]
        actions[index, 3:6] = relative_rotation[:, 0]
        actions[index, 6:9] = relative_rotation[:, 1]
        actions[index, 9] = gripper[index + 1]
    return actions


def validate_camera_calibration(
    calibration: Mapping[str, Any], *, expected_frames: int = OSCAR_NUM_FRAMES
) -> dict[str, Any]:
    """Validate explicit OpenCV world-to-camera calibration without guessing."""

    size = calibration.get("image_size")
    if (
        not isinstance(size, Sequence)
        or isinstance(size, (str, bytes))
        or len(size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size)
    ):
        raise ValueError("camera_image_size_invalid")
    width, height = int(size[0]), int(size[1])
    per_frame = calibration.get("camera_is_per_frame")
    if not isinstance(per_frame, bool):
        raise ValueError("camera_is_per_frame_missing")
    intrinsic = np.asarray(calibration.get("camera_intrinsic"), dtype=np.float64)
    extrinsic = np.asarray(calibration.get("camera_extrinsic"), dtype=np.float64)
    expected_extrinsic_shape = (expected_frames, 4, 4) if per_frame else (4, 4)
    if extrinsic.shape != expected_extrinsic_shape or not np.isfinite(extrinsic).all():
        raise ValueError("camera_extrinsic_shape_invalid")
    allowed_intrinsic_shapes = {(3, 3)}
    if per_frame:
        allowed_intrinsic_shapes.add((expected_frames, 3, 3))
    if intrinsic.shape not in allowed_intrinsic_shapes or not np.isfinite(intrinsic).all():
        raise ValueError("camera_intrinsic_shape_invalid")
    intrinsics = intrinsic[None, :, :] if intrinsic.ndim == 2 else intrinsic
    if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
        raise ValueError("camera_focal_length_invalid")
    if not np.allclose(intrinsics[:, 2, :], np.asarray([0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError("camera_intrinsic_homogeneous_row_invalid")
    extrinsics = extrinsic[None, :, :] if extrinsic.ndim == 2 else extrinsic
    _validate_se3(extrinsics, "camera_extrinsic_not_world_to_camera_se3")
    material = {
        "schema_version": "oscar_camera_calibration.v1",
        "image_size": [width, height],
        "camera_is_per_frame": per_frame,
        "camera_intrinsic": intrinsic.tolist(),
        "camera_extrinsic": extrinsic.tolist(),
        "coordinate_convention": "OpenCV world_or_robot_base_to_camera",
        "guessed": False,
    }
    return {
        **material,
        "camera_calibration_sha256": canonical_sha256(material),
        "camera_intrinsic_array": intrinsic,
        "camera_extrinsic_array": extrinsic,
    }


@dataclass(frozen=True)
class AbsoluteJointOscarConditioningBuilder:
    """Materialize full-resolution first frames and calibrated skeleton videos."""

    calibration_provider: Callable[..., Mapping[str, Any]]
    forward_kinematics: Callable[[np.ndarray], Any]
    skeleton_renderer: Callable[..., Any]
    required_policy_views: tuple[str, ...] = DROID_ROBOARENA_CONCAT_VIEWS
    builder_id: str = "absolute_joint_oscar_conditioning_v1"

    def __call__(
        self,
        *,
        observation: Mapping[str, Any],
        policy_action: Any,
        executed_prefix_steps: int,
        query_index: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        source_paths = observation.get(WAM_SOURCE_VIEW_PATHS)
        if not isinstance(source_paths, Mapping) or set(source_paths) != set(
            self.required_policy_views
        ):
            raise ValueError("full_resolution_wam_source_views_required")
        trajectory = build_absolute_joint_skeleton_trajectory(
            policy_action,
            current_joint_position=observation.get("observation/joint_position"),
            current_gripper_position=observation.get("observation/gripper_position"),
            executed_prefix_steps=executed_prefix_steps,
        )
        joint_frames = trajectory["joint_angles_array"]
        gripper_frames = trajectory["gripper_openness_array"]
        reliability = encode_fk_reliability_actions_10d(
            joint_frames[:17], gripper_frames[:17], forward_kinematics=self.forward_kinematics
        )
        views: dict[str, Any] = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        for view_id in self.required_policy_views:
            source = _safe_file(source_paths[view_id], f"source_view_missing:{view_id}")
            calibration = validate_camera_calibration(
                self.calibration_provider(
                    view_id=view_id,
                    observation=observation,
                    joint_angles=joint_frames,
                    query_index=query_index,
                )
            )
            view_dir = output_dir / view_id.replace("/", "_")
            view_dir.mkdir(parents=True, exist_ok=True)
            first_frame = view_dir / "first_frame.png"
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                if rgb.size != tuple(calibration["image_size"]):
                    raise ValueError(f"camera_calibration_image_size_mismatch:{view_id}")
                rgb.save(first_frame)
            skeleton_path = view_dir / "skeleton.mp4"
            rendered = self.skeleton_renderer(
                view_id=view_id,
                first_frame_path=first_frame,
                joint_angles=joint_frames,
                gripper_openness=gripper_frames,
                camera_intrinsic=calibration["camera_intrinsic_array"],
                camera_extrinsic=calibration["camera_extrinsic_array"],
                camera_is_per_frame=calibration["camera_is_per_frame"],
                fps=OSCAR_FPS,
                output_path=skeleton_path,
            )
            rendered_path = _safe_file(rendered, f"skeleton_renderer_output_missing:{view_id}")
            views[view_id] = {
                "first_frame_path": first_frame,
                "skeleton_video_path": rendered_path,
                "camera_calibration_sha256": calibration["camera_calibration_sha256"],
            }
        return {
            "views": views,
            "reliability_actions_10d": reliability,
            "next_joint_position": trajectory["next_joint_position"],
            "next_gripper_position": trajectory["next_gripper_position"],
            "evidence": {
                "builder_id": self.builder_id,
                "trajectory_sha256": trajectory["trajectory_sha256"],
                "first_frame_source": "recorded_or_prior_same_wam_prediction",
                "physical_future_observation_used": False,
                "joint_targets_clamped": False,
                "reliability_action_semantics": "FK_relative_pose_for_presence_and_timing_only",
                "rendered_skeleton_sha256_by_view": {
                    view_id: file_sha256(Path(view["skeleton_video_path"]))
                    for view_id, view in views.items()
                },
            },
        }


__all__ = [
    "AbsoluteJointOscarConditioningBuilder",
    "OSCAR_FPS",
    "OSCAR_NUM_FRAMES",
    "PANDA_JOINT_LIMITS_RAD",
    "POLICY_ACTION_SHAPE",
    "build_absolute_joint_skeleton_trajectory",
    "encode_fk_reliability_actions_10d",
    "validate_camera_calibration",
]
