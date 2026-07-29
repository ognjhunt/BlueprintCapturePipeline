"""Camera-aligned Franka skeleton conditioning for DROID policy actions.

The builder in this module is deliberately outcome-blind.  It applies a
policy's commanded absolute joint positions to an isolated MuJoCo data object,
projects the resulting Franka kinematic chain into the frozen external and
live wrist cameras, and renders texture-free conditioning videos.  It does not
step task physics, read future RGB, or infer whether the task succeeds.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .droid_oscar_closed_loop_adapter import EXTERIOR_VIEW, WRIST_VIEW
from .droid_policy_bridge import droid_joint_position_action_to_mujoco_targets
from .policy_ranking_thesis import canonical_sha256, file_sha256
from .scene_placement.stance_cameras import link_mounted_camera_spec


SCHEMA_VERSION = "franka_droid_skeleton_conditioning.v1"
TRACE_SCHEMA_VERSION = "franka_droid_skeleton_projection_frame.v1"
DEFAULT_SITE_BASE_TRANSLATION_M = (1.9086943, 0.45, 0.0)
# The local Franka task advances from the base toward +X; the registered site
# places the robot at the near edge of the table facing site +Y.
DEFAULT_SITE_FROM_LOCAL_ROTATION_ROW_MAJOR = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
RUNTIME_EXTERNAL_CAMERA_SPEC = {
    "pos": [1.25, -0.85, 1.10],
    "target": [0.43, 0.16, 0.17],
    "fov": 52.0,
    "up": [0.0, 0.0, 1.0],
}
DEFAULT_BODY_CHAIN = (
    "link0",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger",
)


def _camera_projection(spec: Mapping[str, Any], *, width: int, height: int) -> dict[str, Any]:
    eye = np.asarray(spec["pos"], dtype=np.float64)
    target = np.asarray(spec["target"], dtype=np.float64)
    supplied_up = np.asarray(spec.get("up", (0.0, 0.0, 1.0)), dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    supplied_up /= np.linalg.norm(supplied_up)
    right = np.cross(forward, supplied_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    world_to_camera_rotation = np.stack((right, -up, forward), axis=0)
    vfov = math.radians(float(spec["fov"]))
    focal = float(height) / (2.0 * math.tan(vfov / 2.0))
    return {
        "eye": eye,
        "rotation": world_to_camera_rotation,
        "fx": focal,
        "fy": focal,
        "cx": (float(width) - 1.0) / 2.0,
        "cy": (float(height) - 1.0) / 2.0,
        "width": int(width),
        "height": int(height),
        "spec": {str(key): value for key, value in spec.items()},
    }


def _project(point: Sequence[float], camera: Mapping[str, Any]) -> dict[str, Any]:
    world = np.asarray(point, dtype=np.float64)
    camera_point = np.asarray(camera["rotation"]) @ (world - np.asarray(camera["eye"]))
    z = float(camera_point[2])
    positive_depth = z > 1e-9
    u = float(camera["fx"] * camera_point[0] / z + camera["cx"]) if positive_depth else None
    v = float(camera["fy"] * camera_point[1] / z + camera["cy"]) if positive_depth else None
    in_bounds = bool(
        positive_depth
        and u is not None
        and v is not None
        and 0.0 <= u < int(camera["width"])
        and 0.0 <= v < int(camera["height"])
    )
    return {
        "available": in_bounds,
        "u_px": u,
        "v_px": v,
        "positive_depth": positive_depth,
        "in_image_bounds": in_bounds,
        "camera_position_m": camera_point.tolist(),
    }


def _write_trace(
    path: Path,
    *,
    episode_id: str,
    view_id: str,
    landmark_frames: Sequence[Mapping[str, Sequence[float]]],
    camera_frames: Sequence[Mapping[str, Any]],
    segments: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    if len(landmark_frames) != len(camera_frames):
        raise ValueError("franka_skeleton_camera_frame_count_mismatch")
    rows: list[dict[str, Any]] = []
    minimum_visible = None
    for frame_index, (landmarks, camera) in enumerate(zip(landmark_frames, camera_frames, strict=True)):
        projected = []
        visible = 0
        for landmark_id, position in landmarks.items():
            image_projection = _project(position, camera)
            visible += int(image_projection["available"])
            projected.append(
                {
                    "landmark_id": landmark_id,
                    "reference_position_m": list(map(float, position)),
                    "camera_position_m": image_projection.pop("camera_position_m"),
                    "image_projection": image_projection,
                }
            )
        minimum_visible = visible if minimum_visible is None else min(minimum_visible, visible)
        row: dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "episode_id": episode_id,
            "view_id": view_id,
            "frame_index": frame_index,
            "source_controller_horizon_frame_index": frame_index,
            "source_width": int(camera["width"]),
            "source_height": int(camera["height"]),
            "landmarks": projected,
            "segments": [{"from": start, "to": end} for start, end in segments],
            "projected_landmark_count": visible,
            "physical_future_observation_used": False,
        }
        row["frame_sha256"] = canonical_sha256(row)
        rows.append(row)
    if not rows or int(minimum_visible or 0) < 2:
        raise ValueError(f"franka_skeleton_projection_not_observable:{view_id}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "trace_path": str(path),
        "trace_sha256": file_sha256(path),
        "frame_count": len(rows),
        "minimum_visible_landmarks": int(minimum_visible or 0),
    }


def _render_trace_video(trace_path: Path, output_path: Path, *, width: int, height: int, fps: float) -> dict[str, Any]:
    import cv2

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height))
    )
    if not writer.isOpened():
        raise RuntimeError("franka_skeleton_video_writer_failed")
    nonblank_frames = 0
    try:
        for row in rows:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            points: dict[str, tuple[int, int]] = {}
            for landmark in row["landmarks"]:
                projection = landmark["image_projection"]
                if not projection["available"]:
                    continue
                point = (int(round(projection["u_px"])), int(round(projection["v_px"])))
                points[str(landmark["landmark_id"])] = point
            for segment in row["segments"]:
                start, end = points.get(segment["from"]), points.get(segment["to"])
                if start is not None and end is not None:
                    cv2.line(canvas, start, end, (245, 245, 245), 5, cv2.LINE_AA)
            for landmark_id, point in points.items():
                color = (80, 210, 255) if landmark_id == "hand" else (255, 255, 255)
                cv2.circle(canvas, point, 7 if landmark_id == "hand" else 5, color, -1, cv2.LINE_AA)
            nonblank_frames += int(bool(np.any(canvas)))
            writer.write(canvas)
    finally:
        writer.release()
    if nonblank_frames != len(rows):
        raise ValueError("franka_skeleton_video_contains_blank_frame")
    return {
        "path": str(output_path),
        "sha256": file_sha256(output_path),
        "frame_count": len(rows),
        "nonblank_frame_count": nonblank_frames,
        "texture_free": True,
    }


def _rotation_6d(rotation: np.ndarray) -> np.ndarray:
    first = rotation[:, 0] / np.linalg.norm(rotation[:, 0])
    second = rotation[:, 1] - first * float(np.dot(first, rotation[:, 1]))
    second /= np.linalg.norm(second)
    return np.concatenate((first, second))


@dataclass(frozen=True)
class FrankaDroidSkeletonConditioningBuilder:
    """Concrete conditioning builder accepted by the OSCAR DROID adapter."""

    runtime: Mapping[str, Any]
    camera_contract: Mapping[str, Any]
    site_base_translation_m: tuple[float, float, float] = DEFAULT_SITE_BASE_TRANSLATION_M
    site_from_local_rotation_row_major: tuple[float, ...] = (
        DEFAULT_SITE_FROM_LOCAL_ROTATION_ROW_MAJOR
    )
    num_frames: int = 81
    width: int = 640
    height: int = 480
    fps: float = 15.0

    def __call__(
        self,
        *,
        observation: Mapping[str, Any],
        policy_action: np.ndarray,
        executed_prefix_steps: int,
        query_index: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        mujoco = self.runtime["mujoco"]
        model = self.runtime["model"]
        action = np.asarray(policy_action, dtype=np.float64)
        current = np.asarray(observation["observation/joint_position"], dtype=np.float64)
        current_gripper = float(np.asarray(observation["observation/gripper_position"])[0])
        limits = np.asarray(model.jnt_range[:7], dtype=np.float64)
        mapped = [
            droid_joint_position_action_to_mujoco_targets(row, joint_limits=limits)
            for row in action
        ]
        targets = np.asarray([row["joint_position_target_rad"] for row in mapped])
        grippers = np.asarray([float(np.clip(row[7], 0.0, 1.0)) for row in action])
        knots = np.arange(len(targets) + 1, dtype=np.float64)
        sample_at = np.linspace(0.0, float(len(targets)), int(self.num_frames))
        joint_knots = np.vstack((current, targets))
        gripper_knots = np.concatenate(([current_gripper], grippers))
        joint_samples = np.column_stack(
            [np.interp(sample_at, knots, joint_knots[:, index]) for index in range(7)]
        )
        gripper_samples = np.interp(sample_at, knots, gripper_knots)

        # These are the exact cameras used by ``run_franka_droid_closed_loop``
        # for the articulated robot layer.  The supplied InteriorGS contract
        # remains provenance for the captured background, but its static proxy
        # wrist pose is not substituted for the live policy-observation camera.
        external_spec = dict(RUNTIME_EXTERNAL_CAMERA_SPEC)
        data = mujoco.MjData(model)
        body_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in DEFAULT_BODY_CHAIN
        }
        if min(body_ids.values()) < 0:
            raise RuntimeError("franka_skeleton_body_chain_missing")
        landmark_frames: list[dict[str, list[float]]] = []
        external_cameras: list[dict[str, Any]] = []
        wrist_cameras: list[dict[str, Any]] = []
        hand_positions: list[np.ndarray] = []
        hand_rotations: list[np.ndarray] = []
        for joints, gripper in zip(joint_samples, gripper_samples, strict=True):
            data.qpos[:7] = joints
            data.qpos[7:9] = 0.04 * (1.0 - float(np.clip(gripper, 0.0, 1.0)))
            mujoco.mj_forward(model, data)
            frame = {
                name: np.asarray(data.xpos[body_id], dtype=np.float64).tolist()
                for name, body_id in body_ids.items()
            }
            landmark_frames.append(frame)
            hand_id = body_ids["hand"]
            hand_pos = np.asarray(data.xpos[hand_id], dtype=np.float64)
            hand_rotation = np.asarray(data.xmat[hand_id], dtype=np.float64).reshape(3, 3)
            hand_positions.append(hand_pos)
            hand_rotations.append(hand_rotation)
            wrist_spec = link_mounted_camera_spec(
                parent_translation=hand_pos,
                parent_rotation_row_major=hand_rotation.reshape(-1),
                mount_translation=(0.0, 0.10, 0.03),
                mount_forward=(0.0, 0.0, 1.0),
                mount_up=(0.0, 1.0, 0.0),
                look_distance_m=0.5,
                fov_deg=82.0,
            )
            external_cameras.append(_camera_projection(external_spec, width=self.width, height=self.height))
            wrist_cameras.append(_camera_projection(wrist_spec, width=self.width, height=self.height))

        output_dir.mkdir(parents=True, exist_ok=True)
        arm_chain = DEFAULT_BODY_CHAIN[:9]
        arm_segments = tuple(zip(arm_chain, arm_chain[1:])) + (
            ("hand", "left_finger"),
            ("hand", "right_finger"),
        )
        final_hand_position = hand_positions[-1]
        wrist_action_frames: list[dict[str, list[float]]] = []
        for camera, current_hand in zip(wrist_cameras, hand_positions, strict=True):
            remaining = np.asarray(camera["rotation"]) @ (final_hand_position - current_hand)
            x_offset = float(np.clip(remaining[0], -0.10, 0.10))
            y_offset = float(np.clip(remaining[1], -0.10, 0.10))

            def camera_point_to_world(x: float, y: float, z: float) -> list[float]:
                point = np.asarray(camera["eye"]) + np.asarray(camera["rotation"]).T @ np.asarray(
                    [x, y, z], dtype=np.float64
                )
                return point.tolist()

            wrist_action_frames.append(
                {
                    "wrist_action_center": camera_point_to_world(x_offset, y_offset, 0.25),
                    "wrist_action_x": camera_point_to_world(x_offset + 0.035, y_offset, 0.25),
                    "wrist_action_y": camera_point_to_world(x_offset, y_offset + 0.035, 0.25),
                    "wrist_action_diag": camera_point_to_world(
                        x_offset - 0.025, y_offset - 0.025, 0.25
                    ),
                }
            )
        view_material = {
            EXTERIOR_VIEW: (landmark_frames, external_cameras, arm_segments),
            WRIST_VIEW: (
                wrist_action_frames,
                wrist_cameras,
                (
                    ("wrist_action_center", "wrist_action_x"),
                    ("wrist_action_center", "wrist_action_y"),
                    ("wrist_action_center", "wrist_action_diag"),
                ),
            ),
        }
        views: dict[str, Any] = {}
        trace_evidence: dict[str, Any] = {}
        for view_id, (view_landmarks, camera_frames, view_segments) in view_material.items():
            stem = "external" if view_id == EXTERIOR_VIEW else "wrist"
            trace_path = output_dir / f"query_{query_index:03d}_{stem}_skeleton.jsonl"
            video_path = output_dir / f"query_{query_index:03d}_{stem}_skeleton.mp4"
            trace = _write_trace(
                trace_path,
                episode_id=f"query_{query_index:03d}",
                view_id=view_id,
                landmark_frames=view_landmarks,
                camera_frames=camera_frames,
                segments=view_segments,
            )
            render = _render_trace_video(
                trace_path, video_path, width=self.width, height=self.height, fps=self.fps
            )
            first_path = output_dir / f"query_{query_index:03d}_{stem}_first.png"
            Image.fromarray(np.asarray(observation[view_id], dtype=np.uint8)).save(first_path)
            calibration_material = {
                "view_id": view_id,
                "frozen_camera_contract": self.camera_contract,
                "dynamic_wrist_recomputed_each_frame": view_id == WRIST_VIEW,
                "articulated_layer_camera_source": "franka_droid_runtime_observation_camera",
                "captured_background_camera_contract_is_provenance_not_fk_projection": True,
                "site_base_translation_m": list(self.site_base_translation_m),
                "site_from_local_rotation_row_major": list(
                    self.site_from_local_rotation_row_major
                ),
                "width": self.width,
                "height": self.height,
            }
            views[view_id] = {
                "first_frame_path": first_path,
                "skeleton_video_path": video_path,
                "camera_calibration_sha256": canonical_sha256(calibration_material),
            }
            trace_evidence[view_id] = {**trace, "render": render}

        hand_positions_array = np.asarray(hand_positions)
        translation_delta = np.vstack(
            (np.zeros((1, 3), dtype=np.float64), np.diff(hand_positions_array, axis=0))
        )
        reliability_actions = np.column_stack(
            (
                translation_delta,
                np.asarray([_rotation_6d(rotation) for rotation in hand_rotations]),
                gripper_samples,
            )
        )
        next_index = int(executed_prefix_steps) - 1
        evidence = {
            "schema_version": SCHEMA_VERSION,
            "action_space": "absolute_joint_position",
            "kinematics_source": "pinned_mujoco_franka_model",
            "camera_contract_sha256": canonical_sha256(dict(self.camera_contract)),
            "dynamic_wrist_camera_recomputed_each_frame": True,
            "task_physics_stepped": False,
            "physical_future_observation_used": False,
            "task_outcome_accessed": False,
            "trace_evidence": trace_evidence,
            "claim_boundary": "intended robot motion only; not predicted consequences or success",
        }
        evidence["evidence_sha256"] = canonical_sha256(evidence)
        return {
            "views": views,
            "reliability_actions_10d": reliability_actions,
            "next_joint_position": targets[next_index],
            "next_gripper_position": np.asarray([grippers[next_index]], dtype=np.float64),
            "evidence": evidence,
        }


__all__ = [
    "DEFAULT_SITE_BASE_TRANSLATION_M",
    "FrankaDroidSkeletonConditioningBuilder",
    "SCHEMA_VERSION",
]
