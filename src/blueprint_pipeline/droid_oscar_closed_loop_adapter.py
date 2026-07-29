"""DROID policy-observation adapter for a camera-aligned OSCAR WAM arm."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .droid_policy_bridge import (
    DROID_EXTERIOR_VIEW_1,
    DROID_EXTERIOR_VIEW_2,
    DROID_OPENPI_POLICY_VIEWS,
    DROID_ROBOARENA_CONCAT_VIEWS,
    DROID_WRIST_VIEW,
    validate_droid_action_chunk,
    validate_droid_observation,
)
from .oscar_wam_command_adapter import OSCAR_DEFAULT_NEGATIVE_PROMPT
from .policy_ranking_thesis import canonical_sha256, file_sha256


EXTERIOR_VIEW = DROID_EXTERIOR_VIEW_1
RIGHT_EXTERIOR_VIEW = DROID_EXTERIOR_VIEW_2
WRIST_VIEW = DROID_WRIST_VIEW
REQUIRED_POLICY_VIEWS = DROID_OPENPI_POLICY_VIEWS
ROBOARENA_CONCAT_POLICY_VIEWS = DROID_ROBOARENA_CONCAT_VIEWS


def _safe_file(value: Any, *, reason: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(reason)
    return path


def _load_policy_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        return np.asarray(rgb, dtype=np.uint8)


@dataclass(frozen=True)
class DroidOscarSkeletonTransitionAdapter:
    """Translate one OpenPI DROID action chunk into OSCAR view conditioning.

    ``conditioning_builder`` owns Franka kinematics and calibrated projection.
    It must return a separate first-frame/skeleton-video pair for both required
    policy views, a 16x10 reliability action trace, and commanded-prefix joint
    and gripper state.  The adapter never reads physical future frames.
    """

    conditioning_builder: Callable[..., Mapping[str, Any]]
    action_chunk_rows: int
    required_policy_views: tuple[str, ...] = REQUIRED_POLICY_VIEWS
    adapter_id: str = "droid_oscar_camera_aligned_skeleton_v1"

    def prepare_transition(
        self,
        *,
        observation: Mapping[str, Any],
        policy_action: Any,
        task_prompt: str,
        executed_prefix_steps: int,
        query_index: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        observation_blockers = validate_droid_observation(
            observation, required_views=self.required_policy_views
        )
        if observation_blockers:
            raise ValueError(f"droid_observation_invalid:{observation_blockers[0]}")
        action_blockers = validate_droid_action_chunk(
            policy_action, expected_rows=self.action_chunk_rows
        )
        if action_blockers:
            raise ValueError(f"droid_policy_action_invalid:{action_blockers[0]}")
        if executed_prefix_steps > self.action_chunk_rows:
            raise ValueError("executed_prefix_exceeds_policy_action_chunk")
        built = self.conditioning_builder(
            observation=observation,
            policy_action=np.asarray(policy_action, dtype=np.float64),
            executed_prefix_steps=executed_prefix_steps,
            query_index=query_index,
            output_dir=output_dir,
        )
        views = built.get("views")
        if not isinstance(views, Mapping) or set(views) != set(self.required_policy_views):
            raise ValueError("oscar_conditioning_required_policy_views_mismatch")
        request_views: dict[str, Any] = {}
        for view_id in self.required_policy_views:
            view = views[view_id]
            if not isinstance(view, Mapping):
                raise ValueError(f"oscar_conditioning_view_invalid:{view_id}")
            first_frame = _safe_file(
                view.get("first_frame_path"), reason=f"oscar_first_frame_missing:{view_id}"
            )
            skeleton_video = _safe_file(
                view.get("skeleton_video_path"),
                reason=f"oscar_skeleton_video_missing:{view_id}",
            )
            request_views[view_id] = {
                "first_frame_path": str(first_frame),
                "first_frame_sha256": file_sha256(first_frame),
                "skeleton_video_path": str(skeleton_video),
                "skeleton_video_sha256": file_sha256(skeleton_video),
                "camera_calibration_sha256": str(view.get("camera_calibration_sha256") or ""),
            }
            if len(request_views[view_id]["camera_calibration_sha256"]) != 64:
                raise ValueError(f"oscar_camera_calibration_hash_missing:{view_id}")
        reliability_actions = np.asarray(built.get("reliability_actions_10d"), dtype=float)
        if (
            reliability_actions.ndim != 2
            or reliability_actions.shape[1] != 10
            or not np.isfinite(reliability_actions).all()
        ):
            raise ValueError("oscar_reliability_actions_10d_invalid")
        next_joints = np.asarray(built.get("next_joint_position"), dtype=float)
        next_gripper = np.asarray(built.get("next_gripper_position"), dtype=float)
        if next_joints.shape != (7,) or not np.isfinite(next_joints).all():
            raise ValueError("oscar_commanded_prefix_joint_state_invalid")
        if next_gripper.shape != (1,) or not np.isfinite(next_gripper).all():
            raise ValueError("oscar_commanded_prefix_gripper_state_invalid")
        return {
            "wam_request": {
                "schema_version": "droid_oscar_multiview_request.v1",
                "query_index": query_index,
                "task_prompt": task_prompt,
                "negative_prompt": OSCAR_DEFAULT_NEGATIVE_PROMPT,
                "views": request_views,
                "num_frames": 81,
                "width": 640,
                "height": 480,
                "fps": 15.0,
                "executed_prefix_steps": executed_prefix_steps,
                "source_rgb_context": "one_recorded_or_prior_wam_frame_per_view",
                "required_policy_views": list(self.required_policy_views),
            },
            "reliability_actions_10d": reliability_actions,
            "next_joint_position": next_joints,
            "next_gripper_position": next_gripper,
            "executed_prefix_steps": executed_prefix_steps,
            "conditioning_builder_evidence": dict(built.get("evidence") or {}),
            "physical_future_observation_used": False,
        }

    def advance_policy_observation(
        self,
        *,
        previous_observation: Mapping[str, Any],
        prepared_transition: Mapping[str, Any],
        wam_prediction: Mapping[str, Any],
        executed_prefix_steps: int,
        query_index: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        del previous_observation, executed_prefix_steps, query_index, output_dir
        generated = wam_prediction.get("generated_view_frames")
        if not isinstance(generated, Mapping) or set(generated) != set(
            self.required_policy_views
        ):
            raise ValueError("oscar_prediction_required_policy_views_mismatch")
        observation: dict[str, Any] = {
            "observation/joint_position": np.asarray(
                prepared_transition["next_joint_position"], dtype=np.float64
            ),
            "observation/gripper_position": np.asarray(
                prepared_transition["next_gripper_position"], dtype=np.float64
            ),
            "prompt": str(
                prepared_transition["wam_request"]["task_prompt"]
            ),
        }
        generated_hashes: dict[str, str] = {}
        for view_id in self.required_policy_views:
            frame = _safe_file(
                generated[view_id], reason=f"oscar_generated_view_frame_missing:{view_id}"
            )
            observation[view_id] = _load_policy_image(frame)
            generated_hashes[view_id] = file_sha256(frame)
        blockers = validate_droid_observation(
            observation, required_views=self.required_policy_views
        )
        if blockers:
            raise ValueError(f"advanced_droid_observation_invalid:{blockers[0]}")
        return {
            "observation": observation,
            "provenance": {
                "visual_source": "wam_prediction",
                "state_source": "commanded_prefix_kinematics",
                "physical_future_observation_used": False,
                "generated_view_frame_sha256": generated_hashes,
                "generated_view_count": len(generated_hashes),
                "same_wam_arm_generated_all_views": True,
            },
        }


@dataclass(frozen=True)
class CallableMultiViewOscarWamArm:
    """Run the same frozen OSCAR generator independently for each camera view."""

    generator: Callable[..., Mapping[str, Any]]
    frame_extractor: Callable[..., Path]
    required_policy_views: tuple[str, ...] = REQUIRED_POLICY_VIEWS
    arm_id: str = "oscar_purpose_built_wam_multiview"

    def predict(self, request: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
        views = request.get("views")
        if not isinstance(views, Mapping) or set(views) != set(
            self.required_policy_views
        ):
            raise ValueError("oscar_multiview_request_invalid")
        generated_videos: dict[str, str] = {}
        generated_frames: dict[str, str] = {}
        receipts: dict[str, Any] = {}
        for view_id in self.required_policy_views:
            view_dir = output_dir / view_id.replace("/", "_")
            view_dir.mkdir(parents=True, exist_ok=True)
            receipt = self.generator(
                view_id=view_id,
                view_request=views[view_id],
                task_prompt=request["task_prompt"],
                negative_prompt=request["negative_prompt"],
                output_dir=view_dir,
            )
            video = _safe_file(
                receipt.get("generated_video_path"),
                reason=f"oscar_generated_video_missing:{view_id}",
            )
            frame = self.frame_extractor(
                video_path=video,
                frame_index=int(request.get("executed_prefix_steps") or 8),
                output_path=view_dir / "executed_prefix_frame.png",
            )
            frame = _safe_file(frame, reason=f"oscar_executed_prefix_frame_missing:{view_id}")
            generated_videos[view_id] = str(video)
            generated_frames[view_id] = str(frame)
            receipt_material = {
                str(key): str(value) if isinstance(value, Path) else value
                for key, value in receipt.items()
            }
            receipts[view_id] = {
                "receipt_sha256": canonical_sha256(receipt_material),
                "generated_video_sha256": file_sha256(video),
                "executed_prefix_frame_sha256": file_sha256(frame),
            }
        primary_view = (
            EXTERIOR_VIEW
            if EXTERIOR_VIEW in self.required_policy_views
            else self.required_policy_views[0]
        )
        return {
            "generated_video_path": generated_videos[primary_view],
            "generated_videos_by_view": generated_videos,
            "generated_view_frames": generated_frames,
            "view_receipts": receipts,
            "same_frozen_wam_generated_all_views": True,
            "wam_to_wam_chaining": False,
        }


__all__ = [
    "CallableMultiViewOscarWamArm",
    "DroidOscarSkeletonTransitionAdapter",
    "EXTERIOR_VIEW",
    "RIGHT_EXTERIOR_VIEW",
    "ROBOARENA_CONCAT_POLICY_VIEWS",
    "REQUIRED_POLICY_VIEWS",
    "WRIST_VIEW",
]
