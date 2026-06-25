"""Build a Vast-runnable OSCAR WAM provider runtime bundle.

The bundle is intentionally small: it contains Blueprint's WAM rollout input
manifest, the materialized first-frame and skeleton-conditioning inputs, and a
remote runner that acquires OSCAR source/checkpoint material inside the GPU
runtime before attempting inference.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .oscar_wam_command_adapter import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_WIDTH,
    _materialize_oscar_input_package,
)
from .wam_generated_video_review import (
    validate_generated_mp4_for_review,
    visual_smoke_generated_rollouts_for_review,
)


OSCAR_WAM_PROVIDER_BUNDLE_SCHEMA_VERSION = "oscar_wam_provider_bundle_manifest.v1"
DEFAULT_OSCAR_SOURCE_URL = "https://github.com/wuzy2115/oscar-public.git"
DEFAULT_OSCAR_HF_REPO = "zywu2115/OSCAR-2B"
DEFAULT_BUNDLE_FILENAME = "oscar_wam_provider_runtime_bundle.zip"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def _safe_error_text(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        return type(exc).__name__
    if "/" in text or "\\" in text:
        return Path(text).name or type(exc).__name__
    return text[:240]


def _source_frame_from_wam_generation_step(step_input: Mapping[str, Any]) -> Path:
    candidates = [
        _string(step_input.get("source_policy_observation_frame_path")),
        _string(_mapping(step_input.get("current_policy_observation")).get("camera_frame_path")),
        _string(
            _mapping(
                _mapping(step_input.get("current_policy_observation")).get("visual_observation")
            ).get("camera_frame_path")
        ),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("source_policy_observation_frame_missing")


def _source_action_values(step_input: Mapping[str, Any]) -> list[float]:
    action = _mapping(step_input.get("source_policy_action"))
    values = action.get("action_chunk")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        values = action.get("sonic_latent_action")
        if isinstance(values, Sequence) and values and isinstance(values[0], Sequence):
            values = values[0]
    result: list[float] = []
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        for value in values:
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                continue
    return result


def _task_prompt_from_wam_generation_step(step_input: Mapping[str, Any]) -> str:
    observation = _mapping(step_input.get("current_policy_observation"))
    for key in (
        "language_instruction",
        "task_prompt",
        "task_instruction",
        "instruction",
        "task_description",
    ):
        value = _string(observation.get(key))
        if value:
            return value
    task_id = _string(observation.get("task_id"))
    target_id = _string(observation.get("target_object_id"))
    if task_id or target_id:
        return "Predict the next robot-scene frames after the policy action for " + " ".join(
            part for part in (task_id, target_id) if part
        )
    return "Predict the next robot-scene frames from the Unitree G1 SONIC policy action."


def _materialize_step_first_frame(
    *,
    source_frame: Path,
    output_path: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    frame = cv2.imread(str(source_frame), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("source_policy_observation_frame_decode_failed")
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    ensure_dir(output_path.parent)
    if not cv2.imwrite(str(output_path), resized):
        raise RuntimeError("step_first_frame_write_failed")
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return {
        "path": str(output_path),
        "source_policy_observation_frame_path": str(source_frame),
        "width": width,
        "height": height,
        "source": "wam_generation_step_source_policy_observation_frame",
        "luma_mean": round(float(gray.mean()), 6),
        "luma_range": int(gray.max()) - int(gray.min()),
        "non_dark_fraction": round(float(np.count_nonzero(gray > 12)) / float(width * height), 6),
    }


def _render_step_action_conditioning_video(
    *,
    source_frame: Path,
    action_values: Sequence[float],
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    frame = cv2.imread(str(source_frame), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("source_policy_observation_frame_decode_failed")
    base = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("cv2_video_writer_failed_for_wam_step_action_conditioning")
    values = list(action_values)
    if not values:
        values = [0.0]
    luma_ranges: list[int] = []
    non_dark_fractions: list[float] = []
    action_energy = sum(abs(value) for value in values) / max(len(values), 1)
    left_bias = values[0] if values else 0.0
    right_bias = values[1] if len(values) > 1 else -left_bias
    reach_bias = values[2] if len(values) > 2 else action_energy
    for index in range(max(1, int(num_frames))):
        progress = index / max(int(num_frames) - 1, 1)
        canvas = cv2.convertScaleAbs(base, alpha=0.72, beta=18)
        overlay = canvas.copy()
        target_center = (
            int(width * (0.52 + 0.08 * np.tanh(left_bias + progress * reach_bias))),
            int(height * (0.45 - 0.06 * np.tanh(right_bias + progress * action_energy))),
        )
        cv2.rectangle(
            overlay,
            (target_center[0] - max(18, width // 22), target_center[1] - max(14, height // 28)),
            (target_center[0] + max(18, width // 22), target_center[1] + max(14, height // 28)),
            (32, 190, 220),
            -1,
            cv2.LINE_AA,
        )
        canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
        left_wrist = (
            int(width * (0.30 + 0.08 * progress + 0.03 * np.tanh(left_bias))),
            int(height * (0.70 - 0.18 * progress)),
        )
        right_wrist = (
            int(width * (0.70 - 0.08 * progress + 0.03 * np.tanh(right_bias))),
            int(height * (0.70 - 0.18 * progress)),
        )
        hand_color = (112, 248, 198)
        arm_color = (72, 232, 255)
        target_color = (20, 196, 255)
        for wrist, side in ((left_wrist, -1), (right_wrist, 1)):
            elbow = (wrist[0] - side * int(width * 0.10), wrist[1] + int(height * 0.16))
            cv2.line(canvas, elbow, wrist, arm_color, 5, cv2.LINE_AA)
            cv2.ellipse(
                canvas,
                wrist,
                (max(18, width // 28), max(12, height // 42)),
                0,
                0,
                360,
                hand_color,
                3,
                cv2.LINE_AA,
            )
            palm = (wrist[0] + side * int(width * 0.030), wrist[1] - int(height * 0.020))
            cv2.line(canvas, wrist, palm, (245, 245, 230), 3, cv2.LINE_AA)
            for finger_idx, angle in enumerate((-35, -12, 10, 30)):
                finger_len = int(width * (0.028 - 0.002 * min(finger_idx, 2)))
                dx = int(side * finger_len * np.cos(np.deg2rad(angle)))
                dy = int(finger_len * np.sin(np.deg2rad(angle)) - height * 0.035)
                cv2.line(canvas, palm, (palm[0] + dx, palm[1] + dy), (245, 245, 230), 2, cv2.LINE_AA)
        midpoint = ((left_wrist[0] + right_wrist[0]) // 2, (left_wrist[1] + right_wrist[1]) // 2)
        cv2.arrowedLine(canvas, midpoint, target_center, arm_color, 5, cv2.LINE_AA, tipLength=0.2)
        cv2.circle(canvas, target_center, max(8, width // 55), target_color, 4, cv2.LINE_AA)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        luma_ranges.append(int(gray.max()) - int(gray.min()))
        non_dark_fractions.append(float(np.count_nonzero(gray > 12)) / float(width * height))
        writer.write(canvas)
    writer.release()
    low_signal_blockers: list[str] = []
    if luma_ranges and max(luma_ranges) < 40:
        low_signal_blockers.append("policy_action_conditioning_luma_range_too_low")
    if non_dark_fractions and max(non_dark_fractions) < 0.03:
        low_signal_blockers.append("policy_action_conditioning_foreground_fraction_too_low")
    return {
        "path": str(output_path),
        "frame_count": max(1, int(num_frames)),
        "fps": fps,
        "width": width,
        "height": height,
        "conditioning_mode": "unitree_sonic_policy_action_proxy_over_scene_frame",
        "conditioning_source": "unitree_g1_sonic_policy_action_chunk_over_source_policy_observation",
        "source_policy_observation_frame_path": str(source_frame),
        "source_action_chunk_value_count": len(action_values),
        "source_action_chunk_l1_mean": round(float(action_energy), 6),
        "policy_action_conditioning_proxy_rendered": True,
        "proxy_skeleton_overlay_drawn": False,
        "projected_g1_skeleton_rendered": False,
        "oscar_gripper_scenario_proxy_rendered": True,
        "selected_review_video_background_used": True,
        "visual_signal": {
            "status": "ok" if not low_signal_blockers else "warning_low_signal_proxy_conditioning",
            "blockers": low_signal_blockers,
            "max_luma_range": max(luma_ranges) if luma_ranges else 0,
            "max_non_dark_fraction": round(max(non_dark_fractions), 6)
            if non_dark_fractions
            else 0.0,
        },
    }


def _materialize_oscar_input_package_from_wam_generation_step(
    *,
    step_input: Mapping[str, Any],
    work_dir: Path,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
) -> dict[str, Any]:
    source_frame = _source_frame_from_wam_generation_step(step_input)
    package_dir = work_dir / "oscar_input"
    first_frame = _materialize_step_first_frame(
        source_frame=source_frame,
        output_path=package_dir / "first_frame.png",
        width=width,
        height=height,
    )
    action_values = _source_action_values(step_input)
    skeleton_video = _render_step_action_conditioning_video(
        source_frame=source_frame,
        action_values=action_values,
        output_path=package_dir / "blueprint_proxy_skeleton_conditioning.mp4",
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
    )
    conditioning_validation = validate_generated_mp4_for_review(Path(skeleton_video["path"]))
    conditioning_visual_smoke = visual_smoke_generated_rollouts_for_review(
        rollouts=[
            {
                "rollout_id": "oscar_step_policy_action_conditioning_0001",
                "generated_video_path": skeleton_video["path"],
            }
        ],
        output_dir=work_dir / "oscar_input_conditioning_visual_review",
        generated_at=utc_now_iso(),
    )
    observation = _mapping(step_input.get("current_policy_observation"))
    requested_output = _mapping(step_input.get("requested_output"))
    manifest = {
        "schema_version": "blueprint_oscar_wam_input_package.v1",
        "status": "completed",
        "source_schema_version": _string(step_input.get("schema_version")),
        "step_index": step_input.get("step_index"),
        "first_frame": first_frame,
        "skeleton_video": skeleton_video,
        "conditioning_video_review_validation": conditioning_validation,
        "conditioning_video_visual_smoke": conditioning_visual_smoke,
        "conditioning_video_decode_valid_for_review": conditioning_validation.get("status")
        == "completed",
        "conditioning_video_visually_useful_for_model_input": bool(
            _mapping(conditioning_visual_smoke.get("claim_boundary")).get(
                "visual_rollout_useful_for_task_success_review"
            )
        ),
        "prompt": _task_prompt_from_wam_generation_step(step_input),
        "num_frames": num_frames,
        "fps": fps,
        "height": height,
        "width": width,
        "source_policy_observation_frame_path": str(source_frame),
        "source_action": {
            "action_type": _mapping(step_input.get("source_policy_action")).get("action_type"),
            "action_chunk_value_count": len(action_values),
            "unitree_groot_n17_sonic_action_chunk_present": bool(action_values),
        },
        "requested_output": {
            "next_observation_frame_path": requested_output.get("next_observation_frame_path"),
            "action_conditioned_generation_required": bool(
                requested_output.get("action_conditioned_generation_required")
            ),
        },
        "rgb_video": {
            "path": None,
            "source": "single_policy_observation_frame_only",
            "used_for_oscar_rgb_latent_context": False,
            "rgb_context_mode": "never",
            "omitted_for_wam_generation_step_single_frame_input": True,
        },
        "source_review_video": {
            "camera": _string(
                _mapping(observation.get("visual_observation")).get("camera_id")
            )
            or "head_pov",
            "source": "wam_generation_step_source_policy_observation_frame",
            "review_video_available": False,
            "single_frame_policy_observation_used": True,
        },
        "projected_skeleton_trace": {
            "path": None,
            "available": False,
            "used_for_conditioning": False,
            "row_count": 0,
            "projectable_row_count": 0,
            "conditioning_source": "policy_action_proxy_video_from_unitree_sonic_action_chunk",
            "simulated_state_not_physical_robot_sensor_evidence": True,
        },
        "source_camera": _string(
            _mapping(observation.get("visual_observation")).get("camera_id")
        )
        or "head_pov",
        "task_id": observation.get("task_id"),
        "target_object_id": observation.get("target_object_id"),
        "claim_boundary": {
            "wam_generation_step_input_materialized_for_oscar_provider": True,
            "source_frame_is_simulated_policy_observation": True,
            "single_frame_policy_observation_used_instead_of_review_video": True,
            "policy_action_conditioning_proxy_video_used": True,
            "policy_action_conditioning_proxy_is_not_wam_output": True,
            "policy_action_conditioning_proxy_is_not_physical_robot_sensor_evidence": True,
            "skeleton_conditioning_is_proxy_from_policy_action_chunk": True,
            "projected_g1_skeleton_conditioning_used": False,
            "projected_g1_skeleton_conditioning_is_not_physical_robot_sensor_evidence": True,
            "rgb_video_uses_selected_review_video": False,
            "rgb_video_used_for_oscar_rgb_latent_context": False,
            "rgb_context_mode": "never",
            "true_robot_proprioceptive_skeleton_available": False,
            "generated_input_is_not_model_output": True,
            "conditioning_video_visual_smoke_is_not_wam_output_success": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
    }
    write_json(work_dir / "oscar_wam_input_package_manifest.json", manifest)
    return manifest


def _scrub_local_absolute_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _scrub_local_absolute_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_local_absolute_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub_local_absolute_paths(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return "<local_path_omitted_from_runtime_manifest>"
    return value


def _conditioning_video_input_blockers(input_package: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    validation = _mapping(input_package.get("conditioning_video_review_validation"))
    visual_smoke = _mapping(input_package.get("conditioning_video_visual_smoke"))
    skeleton_video = _mapping(input_package.get("skeleton_video"))
    visual_signal = _mapping(skeleton_video.get("visual_signal"))
    if validation.get("status") != "completed":
        blockers.append("oscar_input_skeleton_conditioning_video_unreadable")
    if input_package.get("conditioning_video_decode_valid_for_review") is not True:
        blockers.append("oscar_input_skeleton_conditioning_video_decode_invalid")
    if visual_smoke.get("status") != "passed_visual_quality_smoke":
        blockers.append("oscar_input_skeleton_conditioning_video_visual_smoke_failed")
    if input_package.get("conditioning_video_visually_useful_for_model_input") is not True:
        blockers.append("oscar_input_skeleton_conditioning_video_not_visually_useful")
    if visual_signal.get("status") == "warning_low_signal_proxy_conditioning":
        blockers.append("oscar_input_skeleton_conditioning_low_signal")
        for blocker in visual_signal.get("blockers", []) or []:
            if isinstance(blocker, str) and blocker:
                blockers.append(f"oscar_input_skeleton_conditioning_{blocker}")
    claim_boundary = _mapping(input_package.get("claim_boundary"))
    projected_trace = _mapping(input_package.get("projected_skeleton_trace"))
    projected_conditioning_used = bool(
        projected_trace.get("used_for_conditioning")
        or claim_boundary.get("projected_g1_skeleton_conditioning_used")
        or skeleton_video.get("projected_g1_skeleton_rendered")
    )
    if projected_conditioning_used:
        projected_path = Path(_string(projected_trace.get("path"))).expanduser()
        if not projected_path.is_file():
            blockers.append("oscar_input_projected_g1_skeleton_trace_missing")
        if int(projected_trace.get("projectable_row_count") or 0) <= 0:
            blockers.append("oscar_input_projected_g1_skeleton_trace_empty")
    return sorted(set(blockers))


def _package_uses_projected_g1_skeleton(input_package: Mapping[str, Any]) -> bool:
    claim_boundary = _mapping(input_package.get("claim_boundary"))
    projected_trace = _mapping(input_package.get("projected_skeleton_trace"))
    skeleton_video = _mapping(input_package.get("skeleton_video"))
    return bool(
        projected_trace.get("used_for_conditioning")
        or claim_boundary.get("projected_g1_skeleton_conditioning_used")
        or skeleton_video.get("projected_g1_skeleton_rendered")
    )


def _runtime_input_package_manifest(
    input_package: Mapping[str, Any],
    *,
    first_frame_runtime_path: str,
    skeleton_runtime_path: str,
    rgb_runtime_path: str | None = None,
    projected_skeleton_runtime_path: str | None = None,
) -> dict[str, Any]:
    try:
        runtime_package = json.loads(json.dumps(dict(input_package)))
    except TypeError:
        runtime_package = dict(input_package)
    runtime_package["first_frame"] = {
        **_mapping(runtime_package.get("first_frame")),
        "path": first_frame_runtime_path,
    }
    runtime_package["skeleton_video"] = {
        **_mapping(runtime_package.get("skeleton_video")),
        "path": skeleton_runtime_path,
    }
    if rgb_runtime_path:
        runtime_package["rgb_video"] = {
            **_mapping(runtime_package.get("rgb_video")),
            "path": rgb_runtime_path,
            "source": _string(_mapping(runtime_package.get("rgb_video")).get("source"))
            or "selected_review_video_rgb_context",
            "used_for_oscar_rgb_latent_context": True,
        }
        runtime_package["source_review_video_path"] = rgb_runtime_path
        source_review_video = _mapping(runtime_package.get("source_review_video"))
        if source_review_video:
            if source_review_video.pop("path", None):
                source_review_video["local_review_video_path_omitted_from_runtime_manifest"] = True
            if "video_path" in source_review_video:
                source_review_video.pop("video_path", None)
                source_review_video["local_review_video_path_omitted_from_runtime_manifest"] = True
            runtime_package["source_review_video"] = source_review_video
    elif runtime_package.pop("source_review_video_path", None):
        runtime_package["local_source_review_video_path_omitted_from_runtime_manifest"] = True
        rgb_video = _mapping(runtime_package.get("rgb_video"))
        if rgb_video:
            if rgb_video.pop("path", None):
                rgb_video["local_rgb_context_path_omitted_from_runtime_manifest"] = True
            rgb_video["used_for_oscar_rgb_latent_context"] = False
            runtime_package["rgb_video"] = rgb_video
        source_review_video = _mapping(runtime_package.get("source_review_video"))
        if source_review_video:
            if source_review_video.pop("path", None):
                source_review_video["local_review_video_path_omitted_from_runtime_manifest"] = True
            if source_review_video.pop("video_path", None):
                source_review_video["local_review_video_path_omitted_from_runtime_manifest"] = True
            runtime_package["source_review_video"] = source_review_video
    if runtime_package.pop("source_mujoco_endpoint_eval_job_dir", None):
        runtime_package["local_source_mujoco_endpoint_eval_job_dir_omitted_from_runtime_manifest"] = True
    projected_trace = _mapping(runtime_package.get("projected_skeleton_trace"))
    if projected_trace:
        if projected_skeleton_runtime_path:
            projected_trace["path"] = projected_skeleton_runtime_path
            projected_trace["runtime_path_rewritten_for_provider_bundle"] = True
        elif projected_trace.pop("path", None):
            projected_trace["local_projected_skeleton_trace_path_omitted_from_runtime_manifest"] = True
        runtime_package["projected_skeleton_trace"] = projected_trace
    validation = _mapping(runtime_package.get("conditioning_video_review_validation"))
    if validation:
        validation["path"] = skeleton_runtime_path
        runtime_package["conditioning_video_review_validation"] = validation
    visual_smoke = _mapping(runtime_package.get("conditioning_video_visual_smoke"))
    if visual_smoke:
        runtime_rollouts: list[dict[str, Any]] = []
        for rollout in visual_smoke.get("rollouts", []) or []:
            if not isinstance(rollout, Mapping):
                continue
            runtime_rollout = dict(rollout)
            runtime_rollout["generated_video_path"] = skeleton_runtime_path
            runtime_samples: list[dict[str, Any]] = []
            for sample in runtime_rollout.get("sampled_frames", []) or []:
                if not isinstance(sample, Mapping):
                    continue
                runtime_sample = dict(sample)
                if runtime_sample.pop("path", None):
                    runtime_sample["local_review_frame_path_omitted_from_runtime_manifest"] = True
                runtime_samples.append(runtime_sample)
            runtime_rollout["sampled_frames"] = runtime_samples
            runtime_rollouts.append(runtime_rollout)
        visual_smoke["rollouts"] = runtime_rollouts
        runtime_package["conditioning_video_visual_smoke"] = visual_smoke
    runtime_package["runtime_paths_rewritten_for_provider_bundle"] = True
    runtime_package["runtime_path_root"] = "BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR"
    rgb_package = _mapping(input_package.get("rgb_video"))
    rgb_context_mode = _string(rgb_package.get("rgb_context_mode")) or "auto"
    projected_conditioning_used = _package_uses_projected_g1_skeleton(input_package)
    projected_conditioning_suppresses_rgb_context = bool(
        projected_conditioning_used and not rgb_runtime_path
    )
    runtime_package["oscar_rgb_context_runtime_contract"] = {
        "rgb_context_packaged": bool(rgb_runtime_path),
        "rgb_context_mode": rgb_context_mode,
        "rgb_context_runtime_path": rgb_runtime_path,
        "remote_runner_appends_rgb_video_arg_when_packaged": bool(rgb_runtime_path),
        "expected_inference_arg": "--rgb-video" if rgb_runtime_path else None,
        "rgb_video_arg_omitted_by_rgb_context_mode": bool(
            rgb_context_mode == "never" and not rgb_runtime_path
        ),
        "projected_g1_skeleton_conditioning_suppresses_rgb_context": projected_conditioning_suppresses_rgb_context,
        "projected_g1_rgb_context_enabled": bool(
            projected_conditioning_used and rgb_runtime_path
        ),
        "raw_secret_values_recorded": False,
    }
    runtime_package["oscar_projected_skeleton_runtime_contract"] = {
        "projected_skeleton_trace_packaged": bool(projected_skeleton_runtime_path),
        "projected_skeleton_trace_runtime_path": projected_skeleton_runtime_path,
        "conditioning_video_can_be_audited_against_projected_skeleton_trace": bool(
            projected_skeleton_runtime_path
        ),
        "raw_secret_values_recorded": False,
    }
    runtime_package["claim_boundary"] = {
        **_mapping(runtime_package.get("claim_boundary")),
        "runtime_manifest_paths_point_to_provider_runtime_inputs": True,
        "local_conditioning_review_frame_paths_omitted_from_runtime_manifest": True,
        "rgb_video_uses_selected_review_video": bool(rgb_runtime_path),
        "rgb_context_packaging_is_input_contract_not_rollout_quality_proof": True,
        "projected_g1_skeleton_trace_packaging_is_input_provenance_not_rollout_quality_proof": True,
        "projected_g1_skeleton_conditioning_suppresses_rgb_context": projected_conditioning_suppresses_rgb_context,
        "projected_g1_rgb_context_enabled": bool(
            projected_conditioning_used and rgb_runtime_path
        ),
    }
    return _scrub_local_absolute_paths(runtime_package)


def _runtime_rollout_manifest(
    rollout_manifest: Mapping[str, Any],
    *,
    projected_skeleton_runtime_path: str | None = None,
) -> dict[str, Any]:
    try:
        runtime_manifest = json.loads(json.dumps(dict(rollout_manifest)))
    except TypeError:
        runtime_manifest = dict(rollout_manifest)
    if runtime_manifest.pop("source_mujoco_endpoint_eval_job_dir", None):
        runtime_manifest[
            "local_source_mujoco_endpoint_eval_job_dir_omitted_from_runtime_manifest"
        ] = True
    inputs = _mapping(runtime_manifest.get("inputs"))
    if inputs:
        if projected_skeleton_runtime_path:
            inputs["g1_projected_skeleton_trace_jsonl"] = projected_skeleton_runtime_path
        elif inputs.pop("g1_projected_skeleton_trace_jsonl", None):
            inputs[
                "local_g1_projected_skeleton_trace_jsonl_omitted_from_runtime_manifest"
            ] = True
        if inputs.pop("g1_projected_skeleton_manifest", None):
            inputs[
                "local_g1_projected_skeleton_manifest_omitted_from_runtime_manifest"
            ] = True
        runtime_manifest["inputs"] = inputs
    runtime_manifest["runtime_paths_rewritten_for_provider_bundle"] = True
    runtime_manifest["runtime_path_root"] = "BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR"
    return _scrub_local_absolute_paths(runtime_manifest)


def _materialized_package_from_existing(
    *,
    oscar_input_dir: Path,
    package_manifest_path: Path | None,
    rollout_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    first_frame = oscar_input_dir / "first_frame.png"
    skeleton = oscar_input_dir / "blueprint_proxy_skeleton_conditioning.mp4"
    if not first_frame.is_file():
        raise FileNotFoundError("oscar_input_first_frame_missing")
    if not skeleton.is_file():
        raise FileNotFoundError("oscar_input_skeleton_conditioning_video_missing")
    source_manifest_path = package_manifest_path
    if source_manifest_path is None:
        candidate = oscar_input_dir.parent / "oscar_wam_input_package_manifest.json"
        source_manifest_path = candidate if candidate.is_file() else None
    if source_manifest_path and source_manifest_path.is_file():
        manifest = _read_json(source_manifest_path)
    else:
        manifest = {
            "schema_version": "blueprint_oscar_wam_input_package.v1",
            "status": "completed",
            "prompt": "Predict the next robot-scene frames from Blueprint action conditioning.",
            "num_frames": DEFAULT_NUM_FRAMES,
            "fps": DEFAULT_FPS,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "source_mujoco_endpoint_eval_job_dir": rollout_manifest.get(
                "source_mujoco_endpoint_eval_job_dir"
            ),
        }
    manifest["first_frame"] = {**_mapping(manifest.get("first_frame")), "path": str(first_frame)}
    manifest["skeleton_video"] = {**_mapping(manifest.get("skeleton_video")), "path": str(skeleton)}
    conditioning_validation = validate_generated_mp4_for_review(skeleton)
    conditioning_visual_smoke = visual_smoke_generated_rollouts_for_review(
        rollouts=[
            {
                "rollout_id": "oscar_conditioning_video_0001",
                "generated_video_path": str(skeleton),
            }
        ],
        output_dir=oscar_input_dir.parent / "oscar_input_conditioning_visual_review",
        generated_at=utc_now_iso(),
    )
    manifest["conditioning_video_review_validation"] = conditioning_validation
    manifest["conditioning_video_visual_smoke"] = conditioning_visual_smoke
    manifest["conditioning_video_decode_valid_for_review"] = (
        conditioning_validation.get("status") == "completed"
    )
    manifest["conditioning_video_visually_useful_for_model_input"] = bool(
        _mapping(conditioning_visual_smoke.get("claim_boundary")).get(
            "visual_rollout_useful_for_task_success_review"
        )
    )
    manifest["claim_boundary"] = {
        **_mapping(manifest.get("claim_boundary")),
        "skeleton_conditioning_is_proxy_from_mujoco_trace": True,
        "true_robot_proprioceptive_skeleton_available": False,
        "generated_input_is_not_model_output": True,
        "conditioning_video_visual_smoke_is_not_wam_output_success": True,
    }
    return manifest


REMOTE_RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import signal
import textwrap
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "wam_runtime_result.v1"
OSCAR_SOURCE_URL = os.environ.get("BLUEPRINT_OSCAR_WAM_SOURCE_URL", "https://github.com/wuzy2115/oscar-public.git")
OSCAR_HF_REPO = os.environ.get("BLUEPRINT_OSCAR_WAM_HF_REPO", "zywu2115/OSCAR-2B")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase(name: str, **fields: Any) -> None:
    payload = {
        "phase": name,
        "observed_at_epoch": round(time.time(), 3),
        **fields,
        "raw_secret_values_recorded": False,
    }
    print("BLUEPRINT_WAM_RUNTIME_PHASE:" + json.dumps(payload, sort_keys=True), flush=True)


def _redacted_tail(text: str, *, limit: int = 4000) -> str:
    if not text:
        return ""
    redacted = text[-limit:]
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            redacted = redacted.replace(value, "<redacted-secret>")
    return redacted


def _validate_generated_video(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": "wam_generated_video_review_validation.v1",
            "status": "blocked",
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "blockers": ["generated_video_missing"],
        }
    try:
        import cv2
    except Exception as exc:
        return {
            "schema_version": "wam_generated_video_review_validation.v1",
            "status": "blocked",
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "blockers": [f"opencv_import_failed:{type(exc).__name__}"],
        }
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {
            "schema_version": "wam_generated_video_review_validation.v1",
            "status": "blocked",
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "blockers": ["generated_video_unreadable"],
        }
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        blockers: list[str] = []
        if frame_count <= 0:
            blockers.append("generated_video_frame_count_unavailable")
        if width <= 0 or height <= 0:
            blockers.append("generated_video_dimensions_unavailable")
        sampled_frames = []
        readable = 0
        if frame_count > 0:
            indices = sorted({0, max(0, frame_count // 2), max(0, frame_count - 1)})
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                readable += 1
                sampled_frames.append(
                    {
                        "frame_index": int(frame_index),
                        "height": int(frame.shape[0]),
                        "width": int(frame.shape[1]),
                    }
                )
        if frame_count > 0 and readable <= 0:
            blockers.append("generated_video_sample_frames_unreadable")
        return {
            "schema_version": "wam_generated_video_review_validation.v1",
            "status": "completed" if not blockers else "blocked",
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "frame_count": frame_count,
            "fps": round(fps, 6),
            "width": width,
            "height": height,
            "readable_sampled_frame_count": readable,
            "sampled_frames": sampled_frames,
            "blockers": blockers,
        }
    finally:
        capture.release()


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 3600,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    _phase(
        "subprocess_started",
        argv0=Path(argv[0]).name if argv else "",
        argv1=argv[1] if len(argv) > 1 and not argv[1].startswith("-") else "",
        cwd=str(cwd) if cwd else None,
        timeout_seconds=timeout,
    )
    completed = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=dict(env) if env is not None else None,
    )
    _phase(
        "subprocess_completed",
        argv0=Path(argv[0]).name if argv else "",
        returncode=completed.returncode,
        duration_seconds=round(time.monotonic() - started, 6),
    )
    return {
        "argv_redacted": [
            "<hf_token_env>" if "HF_TOKEN" in item or item.startswith("--token") else item
            for item in argv
        ],
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout_size_bytes": len(completed.stdout or ""),
        "stderr_size_bytes": len(completed.stderr or ""),
        "stdout_tail_redacted": _redacted_tail(completed.stdout or ""),
        "stderr_tail_redacted": _redacted_tail(completed.stderr or ""),
        "raw_secret_values_recorded": False,
    }


def _dedupe_existing_dirs(paths: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in paths:
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _cuda_library_path_candidates() -> list[str]:
    candidates = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    try:
        import torch

        torch_package_dir = Path(torch.__file__).resolve().parent
        candidates.append(str(torch_package_dir / "lib"))
        site_root = torch_package_dir.parent
        candidates.extend(str(path) for path in (site_root / "nvidia").glob("*/lib"))
        candidates.extend(str(path) for path in (site_root / "nvidia").glob("*/lib64"))
    except Exception:
        pass
    candidates.extend(part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part)
    return _dedupe_existing_dirs(candidates)


def _prepare_cuda_library_env(work_dir: Path, base_env: Mapping[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
    library_dirs = _cuda_library_path_candidates()
    shim_dir = work_dir / "cuda_lib_shims"
    shim_dir.mkdir(parents=True, exist_ok=True)
    unversioned = None
    versioned = None
    for directory in library_dirs:
        candidate = Path(directory) / "libcudart.so"
        if candidate.exists():
            unversioned = candidate
            break
        matches = sorted(Path(directory).glob("libcudart.so*"))
        if matches and versioned is None:
            versioned = matches[0]
    shim_path = shim_dir / "libcudart.so"
    shim_created = False
    shim_target = None
    if unversioned is None and versioned is not None:
        try:
            if shim_path.exists() or shim_path.is_symlink():
                shim_path.unlink()
            shim_path.symlink_to(versioned)
            shim_created = True
            shim_target = str(versioned)
        except Exception:
            try:
                shutil.copy2(versioned, shim_path)
                shim_created = True
                shim_target = str(versioned)
            except Exception:
                shim_created = False
    env = dict(base_env)
    existing_ld = [part for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    ld_paths = _dedupe_existing_dirs([str(shim_dir), *library_dirs, *existing_ld])
    if ld_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)
    return env, {
        "status": "completed",
        "library_dirs": library_dirs,
        "shim_dir": str(shim_dir),
        "libcudart_unversioned_present": unversioned is not None,
        "libcudart_versioned_source": str(versioned) if versioned is not None else None,
        "libcudart_shim_created": shim_created,
        "libcudart_shim_target": shim_target,
        "ld_library_path_configured": bool(ld_paths),
        "raw_secret_values_recorded": False,
    }


def _python_probe(candidate: str) -> dict[str, Any]:
    path = shutil.which(candidate) or candidate
    if not path or not Path(path).exists():
        return {
            "candidate": candidate,
            "path": path,
            "exists": False,
        }
    code = (
        "import importlib.util, json, sys\n"
        "payload={'executable': sys.executable, 'pip_importable': importlib.util.find_spec('pip') is not None, "
        "'torch_importable': False, 'torch_cuda_available': False, 'cuda_device_count': 0}\n"
        "try:\n"
        " import torch\n"
        " payload['torch_importable']=True\n"
        " payload['torch_version']=getattr(torch, '__version__', None)\n"
        " payload['torch_cuda_available']=bool(torch.cuda.is_available())\n"
        " payload['cuda_device_count']=int(torch.cuda.device_count())\n"
        "except Exception as exc:\n"
        " payload['torch_error_type']=type(exc).__name__\n"
        "print(json.dumps(payload))\n"
    )
    try:
        completed = subprocess.run(
            [path, "-c", code],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except Exception as exc:
        return {
            "candidate": candidate,
            "path": path,
            "exists": True,
            "probe_error_type": type(exc).__name__,
        }
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        payload = {}
    return {
        "candidate": candidate,
        "path": path,
        "exists": True,
        "returncode": completed.returncode,
        "payload": payload,
    }


def _find_python() -> tuple[str, dict[str, Any]]:
    configured = os.environ.get("BLUEPRINT_WAM_PROVIDER_PYTHON", "").strip()
    if configured:
        probe = _python_probe(configured)
        return configured, {
            "source": "configured_python",
            "selected_python": configured,
            "selected_probe": probe,
            "candidate_probes": [probe],
        }
    candidates = [
        sys.executable,
        "python",
        "/opt/conda/bin/python",
        "/usr/local/bin/python",
        "python3",
        "/usr/bin/python3",
    ]
    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = shutil.which(candidate) or candidate
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        probes.append(_python_probe(candidate))
    usable = [
        probe
        for probe in probes
        if probe.get("exists") is True and probe.get("returncode") == 0
    ]
    if usable:
        usable.sort(
            key=lambda probe: (
                _mapping(probe.get("payload")).get("torch_cuda_available") is True,
                _mapping(probe.get("payload")).get("torch_importable") is True,
                _mapping(probe.get("payload")).get("pip_importable") is True,
            ),
            reverse=True,
        )
        selected = usable[0]
        selected_python = _string(_mapping(selected.get("payload")).get("executable")) or _string(
            selected.get("path")
        )
        return selected_python, {
            "source": "auto_python_probe",
            "selected_python": selected_python,
            "selected_probe": selected,
            "candidate_probes": probes,
        }
    return sys.executable, {
        "source": "fallback_sys_executable",
        "selected_python": sys.executable,
        "candidate_probes": probes,
    }


def _bootstrap_python(work_dir: Path) -> tuple[str, dict[str, Any]]:
    base_python, selection_detail = _find_python()
    configured = os.environ.get("BLUEPRINT_WAM_PROVIDER_VENV_PYTHON", "").strip()
    if configured and Path(configured).is_file():
        return configured, {
            "status": "completed",
            "source": "configured_venv_python",
            "base_python": base_python,
            "python": configured,
            "python_selection": selection_detail,
        }
    disable_venv = os.environ.get("BLUEPRINT_WAM_PROVIDER_DISABLE_VENV", "true").strip().lower()
    if disable_venv in {"1", "true", "yes", "on"}:
        return base_python, {
            "status": "completed",
            "source": "base_python_venv_disabled",
            "base_python": base_python,
            "python": base_python,
            "python_selection": selection_detail,
        }
    venv_dir = work_dir / ".blueprint_wam_venv"
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.is_file():
        detail = _run(
            [base_python, "-m", "venv", "--system-site-packages", str(venv_dir)],
            timeout=300,
        )
        if detail.get("returncode") != 0 or not venv_python.is_file():
            return base_python, {
                "status": "completed",
                "source": "base_python_after_venv_create_failed",
                "base_python": base_python,
                "fallback_python": base_python,
                "python": base_python,
                "venv_dir": str(venv_dir),
                "venv_create_failed_nonfatal": True,
                "python_selection": selection_detail,
                "venv_subprocess": detail,
            }
    return str(venv_python), {
        "status": "completed",
        "source": "venv_with_system_site_packages",
        "base_python": base_python,
        "python": str(venv_python),
        "venv_dir": str(venv_dir),
        "python_selection": selection_detail,
    }


def _clone_source(work_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    configured = os.environ.get("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", "").strip()
    if configured and (Path(configured) / "inference" / "inference_oscar.py").is_file():
        return Path(configured).resolve(), {
            "status": "completed",
            "source": "configured_path",
            "path": str(Path(configured).resolve()),
        }
    target = work_dir / "external" / "oscar-public"
    if (target / "inference" / "inference_oscar.py").is_file():
        return target, {"status": "completed", "source": "existing_cache", "path": str(target)}
    if not shutil.which("git"):
        return None, {"status": "blocked", "blockers": ["git_missing_for_oscar_source_clone"]}
    target.parent.mkdir(parents=True, exist_ok=True)
    detail = _run(["git", "clone", "--depth", "1", OSCAR_SOURCE_URL, str(target)], timeout=900)
    blockers = []
    if detail["returncode"] != 0:
        blockers.append("oscar_source_clone_failed")
    if not (target / "inference" / "inference_oscar.py").is_file():
        blockers.append("oscar_inference_entrypoint_missing_after_clone")
    return (target if not blockers else None), {
        "status": "completed" if not blockers else "blocked",
        "source": "git_clone",
        "path": str(target),
        "blockers": blockers,
        "subprocess": detail,
    }


def _python_env_for_source(source_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if source_root is not None:
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(source_root)
            if not existing_pythonpath
            else str(source_root) + os.pathsep + existing_pythonpath
        )
    return env


def _write_text_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _apply_oscar_source_compatibility(source_root: Path) -> dict[str, Any]:
    strategy = os.environ.get(
        "BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY",
        "torch_sdpa_compat_shim",
    ).strip() or "torch_sdpa_compat_shim"
    if strategy in {"require_real_transformer_engine", "none", "disabled"}:
        removed_paths: list[str] = []
        if strategy in {"none", "disabled"}:
            for candidate in [
                source_root / "transformer_engine",
                source_root / "transformer_engine-2.0.0.dist-info",
            ]:
                if not candidate.exists():
                    continue
                marker_path = (
                    candidate / "__init__.py"
                    if candidate.is_dir() and candidate.name == "transformer_engine"
                    else candidate / "METADATA"
                )
                marker_text = (
                    marker_path.read_text(encoding="utf-8", errors="ignore")
                    if marker_path.is_file()
                    else ""
                )
                if "BLUEPRINT_COMPAT_SHIM" in marker_text or "Blueprint OSCAR" in marker_text:
                    if candidate.is_dir():
                        shutil.rmtree(candidate)
                    else:
                        candidate.unlink()
                    removed_paths.append(str(candidate))
        return {
            "status": "skipped",
            "strategy": strategy,
            "files_written": [],
            "compat_shim_paths_removed": removed_paths,
            "raw_secret_values_recorded": False,
        }
    shim_root = source_root / "transformer_engine"
    files = {
        shim_root / "__init__.py": """
# Blueprint-local TransformerEngine compatibility shim for OSCAR inference.
# This is only written when the real transformer_engine package is not required.
from . import common
from . import pytorch

BLUEPRINT_COMPAT_SHIM = True
""",
        shim_root / "common" / "__init__.py": """
from . import recipe

BLUEPRINT_COMPAT_SHIM = True
""",
        shim_root / "common" / "recipe.py": """
from __future__ import annotations

import enum
from typing import Any

BLUEPRINT_COMPAT_SHIM = True


class Format(enum.Enum):
    E4M3 = "e4m3"
    HYBRID = "hybrid"


class _Recipe:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class DelayedScaling(_Recipe):
    pass


class Float8CurrentScaling(_Recipe):
    pass


class Float8BlockScaling(_Recipe):
    pass


class MXFP8BlockScaling(_Recipe):
    pass


class NVFP4BlockScaling(_Recipe):
    pass


class CustomRecipe(_Recipe):
    pass
""",
        shim_root / "pytorch" / "__init__.py": """
# PyTorch SDPA fallback surface for OSCAR's optional TransformerEngine imports.
import torch

from . import distributed, ops
from .attention import DotProductAttention, apply_rotary_pos_emb
from .fp8 import FP8GlobalStateManager, fp8_autocast, fp8_model_init, quantized_model_init
from .float8_tensor import Float8Tensor
from .tensor import QuantizedTensor

BLUEPRINT_COMPAT_SHIM = True
RMSNorm = torch.nn.RMSNorm


class Linear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, *args, bias: bool = True, return_bias: bool = False, **kwargs) -> None:
        del args
        init_method = kwargs.pop("init_method", None)
        self.return_bias = bool(return_bias)
        self.use_bias = bool(bias)
        self.parallel_mode = kwargs.pop("parallel_mode", None)
        self.tp_size = int(kwargs.pop("tp_size", 1) or 1)
        kwargs.pop("sequence_parallel", None)
        kwargs.pop("fuse_wgrad_accumulation", None)
        kwargs.pop("tp_group", None)
        kwargs.pop("get_rng_state_tracker", None)
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        if callable(init_method):
            init_method(self.weight)

    def forward(self, input, *args, **kwargs):
        del args, kwargs
        out = super().forward(input)
        return (out, self.bias) if self.return_bias else out

    def set_tensor_parallel_group(self, *args, **kwargs) -> None:
        del args, kwargs
        return None

    def backward_dw(self) -> None:
        return None


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, hidden_size=None, normalized_shape=None, *args, eps: float = 1e-5, **kwargs) -> None:
        del args, kwargs
        super().__init__(normalized_shape or hidden_size, eps=eps)


class LayerNormLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, *args, eps: float = 1e-5, bias: bool = True, return_bias: bool = False, **kwargs) -> None:
        super().__init__()
        del args
        init_method = kwargs.pop("init_method", None)
        kwargs.pop("sequence_parallel", None)
        kwargs.pop("fuse_wgrad_accumulation", None)
        kwargs.pop("tp_group", None)
        kwargs.pop("tp_size", None)
        kwargs.pop("get_rng_state_tracker", None)
        kwargs.pop("parallel_mode", None)
        kwargs.pop("return_layernorm_output", None)
        kwargs.pop("zero_centered_gamma", None)
        kwargs.pop("normalization", None)
        self.return_bias = bool(return_bias)
        self.use_bias = bool(bias)
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = 1
        self.layer_norm = torch.nn.LayerNorm(in_features, eps=eps)
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        if callable(init_method):
            init_method(self.linear.weight)

    def forward(self, input, *args, **kwargs):
        del args, kwargs
        out = self.linear(self.layer_norm(input))
        return (out, self.bias) if self.return_bias else out

    def set_tensor_parallel_group(self, *args, **kwargs) -> None:
        del args, kwargs
        return None

    def backward_dw(self) -> None:
        return None


class GroupedLinear(Linear):
    pass
""",
        shim_root / "pytorch" / "ops" / "__init__.py": """
from __future__ import annotations

from typing import Any

import torch

BLUEPRINT_COMPAT_SHIM = True


class FusibleOperation(torch.nn.Module):
    def forward(self, x, *args: Any, **kwargs: Any):
        del args, kwargs
        return x


class Sequential(torch.nn.Sequential):
    pass


class _Activation(FusibleOperation):
    fn = staticmethod(lambda x: x)

    def forward(self, x, *args: Any, **kwargs: Any):
        del args, kwargs
        return self.fn(x)


class GELU(_Activation):
    fn = staticmethod(torch.nn.functional.gelu)


class ReLU(_Activation):
    fn = staticmethod(torch.nn.functional.relu)


class SiLU(_Activation):
    fn = staticmethod(torch.nn.functional.silu)


class SwiGLU(FusibleOperation):
    pass


class GEGLU(FusibleOperation):
    pass


class ReGLU(FusibleOperation):
    pass


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, norm_shape, *args: Any, eps: float = 1e-5, **kwargs: Any) -> None:
        del args, kwargs
        super().__init__(norm_shape, eps=eps)


class RMSNorm(torch.nn.RMSNorm):
    pass


class BasicLinear(torch.nn.Linear):
    pass


class Bias(FusibleOperation):
    def __init__(self, size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        self.bias = torch.nn.Parameter(torch.zeros(size, device=device, dtype=dtype))

    def forward(self, x, *args: Any, **kwargs: Any):
        del args, kwargs
        return x + self.bias


class ReduceScatter(FusibleOperation):
    pass


class AllReduce(FusibleOperation):
    pass
""",
        shim_root / "pytorch" / "distributed" / "__init__.py": """
from __future__ import annotations

from typing import Any

BLUEPRINT_COMPAT_SHIM = True


def activation_recompute_forward(*args: Any, **kwargs: Any):
    del kwargs
    if args and callable(args[0]):
        return args[0](*args[1:])
    return None


def get_all_rng_states(*args: Any, **kwargs: Any) -> dict:
    del args, kwargs
    return {}


def checkpoint(function, *args: Any, **kwargs: Any):
    return function(*args, **kwargs)


class CudaRNGStatesTracker:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def add(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    def fork(self, *args: Any, **kwargs: Any):
        del args, kwargs
        from contextlib import nullcontext

        return nullcontext()
""",
        shim_root / "pytorch" / "module" / "__init__.py": """
BLUEPRINT_COMPAT_SHIM = True
""",
        shim_root / "pytorch" / "module" / "base.py": """
from __future__ import annotations

import torch

BLUEPRINT_COMPAT_SHIM = True


class TransformerEngineBaseModule(torch.nn.Module):
    pass


def get_dummy_wgrad(*args, **kwargs):
    del args, kwargs
    return None


def get_workspace(*args, **kwargs):
    del args, kwargs
    return None
""",
        shim_root / "pytorch" / "fp8.py": """
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

BLUEPRINT_COMPAT_SHIM = True


def fp8_autocast(*args: Any, **kwargs: Any):
    del args, kwargs
    return nullcontext()


def fp8_model_init(*args: Any, **kwargs: Any):
    del args, kwargs
    return nullcontext()


def quantized_model_init(*args: Any, **kwargs: Any):
    del args, kwargs
    return nullcontext()


class FP8GlobalStateManager:
    @staticmethod
    def is_fp8_enabled() -> bool:
        return False

    @staticmethod
    def get_fp8_recipe() -> None:
        return None

    @staticmethod
    def get_fp8_group() -> None:
        return None

    @staticmethod
    def is_first_fp8_module() -> bool:
        return False

    @staticmethod
    def add_fp8_tensors_to_global_buffer(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    @staticmethod
    def reduce_and_update_fp8_tensors(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    @staticmethod
    def set_skip_fp8_weight_update_tensor(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None
""",
        shim_root / "pytorch" / "float8_tensor.py": """
from .tensor.float8_tensor import Float8Tensor

BLUEPRINT_COMPAT_SHIM = True
""",
        shim_root / "pytorch" / "tensor" / "__init__.py": """
from __future__ import annotations

from typing import Any

import torch

BLUEPRINT_COMPAT_SHIM = True


class QuantizedTensor:
    def __init__(self, data: torch.Tensor | None = None, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._data = data

    def dequantize(self) -> torch.Tensor:
        if self._data is None:
            raise RuntimeError("QuantizedTensor shim has no backing tensor")
        return self._data

    def from_float8(self) -> torch.Tensor:
        return self.dequantize()


from .float8_tensor import Float8Tensor
from .mxfp8_tensor import MXFP8Tensor

__all__ = ["Float8Tensor", "MXFP8Tensor", "QuantizedTensor"]
""",
        shim_root / "pytorch" / "tensor" / "float8_tensor.py": """
from __future__ import annotations

from typing import Any

import torch

from . import QuantizedTensor

BLUEPRINT_COMPAT_SHIM = True


class Float8Tensor(QuantizedTensor):
    @classmethod
    def make_like(cls, tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        return tensor
""",
        shim_root / "pytorch" / "tensor" / "float8_blockwise_tensor.py": """
from .float8_tensor import Float8Tensor

BLUEPRINT_COMPAT_SHIM = True


class Float8BlockwiseQTensor(Float8Tensor):
    pass
""",
        shim_root / "pytorch" / "tensor" / "mxfp8_tensor.py": """
from .float8_tensor import Float8Tensor

BLUEPRINT_COMPAT_SHIM = True


class MXFP8Tensor(Float8Tensor):
    pass
""",
        shim_root / "pytorch" / "tensor" / "utils.py": """
from __future__ import annotations

from typing import Any

import torch

BLUEPRINT_COMPAT_SHIM = True


def replace_raw_data(fp8_tensor: Any, new_raw_data: torch.Tensor) -> None:
    if hasattr(fp8_tensor, "_data"):
        fp8_tensor._data = new_raw_data


def cast_master_weights_to_fp8(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    return None


def post_all_gather_processing(tensor: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return tensor
""",
        shim_root / "pytorch" / "attention" / "__init__.py": """
# Minimal TransformerEngine attention shim backed by torch SDPA.
from __future__ import annotations

from typing import Any

import torch
from torch import nn

BLUEPRINT_COMPAT_SHIM = True


def _flatten_bshd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])


class DotProductAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int | None = None,
        kv_channels: int | None = None,
        *,
        attention_dropout: float = 0.0,
        qkv_format: str = "bshd",
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = kv_channels
        self.attention_dropout = float(attention_dropout or 0.0)
        self.qkv_format = qkv_format

    def set_context_parallel_group(self, *_: Any, **__: Any) -> None:
        return None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError("DotProductAttention shim expects q/k/v rank-4 tensors")
        if self.qkv_format == "sbhd":
            q_bhsd = query.permute(1, 2, 0, 3)
            k_bhsd = key.permute(1, 2, 0, 3)
            v_bhsd = value.permute(1, 2, 0, 3)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            return out.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], -1)
        q_bhsd = query.permute(0, 2, 1, 3)
        k_bhsd = key.permute(0, 2, 1, 3)
        v_bhsd = value.permute(0, 2, 1, 3)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        return _flatten_bshd(out.permute(0, 2, 1, 3).contiguous())


def apply_rotary_pos_emb(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    *,
    tensor_format: str = "bshd",
    fused: bool = True,
) -> torch.Tensor:
    del fused
    if tensor.dim() != 4:
        raise ValueError("apply_rotary_pos_emb shim expects a rank-4 tensor")
    half = tensor.shape[-1] // 2
    freqs = freqs.to(device=tensor.device, dtype=torch.float32)
    if freqs.shape[-1] >= tensor.shape[-1]:
        freqs = freqs[..., :half]
    if freqs.shape[-1] != half:
        raise ValueError(f"rotary freqs last dim {freqs.shape[-1]} does not match half head dim {half}")
    while freqs.dim() > 2 and freqs.shape[1] == 1:
        freqs = freqs.squeeze(1)
    while freqs.dim() > 2 and freqs.shape[-2] == 1:
        freqs = freqs.squeeze(-2)
    if freqs.dim() != 2:
        freqs = freqs.reshape(freqs.shape[0], half)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    if tensor_format == "sbhd":
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]
    else:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    even = tensor[..., 0::2].to(torch.float32)
    odd = tensor[..., 1::2].to(torch.float32)
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(tensor.dtype)
""",
        shim_root / "pytorch" / "attention" / "rope.py": """
from . import apply_rotary_pos_emb

BLUEPRINT_COMPAT_SHIM = True
""",
        source_root / "transformer_engine-2.0.0.dist-info" / "METADATA": """
Metadata-Version: 2.1
Name: transformer-engine
Version: 2.0.0
Summary: Blueprint OSCAR PyTorch SDPA compatibility shim metadata
""",
        source_root / "transformer_engine-2.0.0.dist-info" / "WHEEL": """
Wheel-Version: 1.0
Generator: blueprint-oscar-wam-provider
Root-Is-Purelib: true
Tag: py3-none-any
""",
        source_root / "transformer_engine-2.0.0.dist-info" / "top_level.txt": """
transformer_engine
""",
    }
    written: list[str] = []
    for path, content in files.items():
        changed = _write_text_if_changed(path, textwrap.dedent(content).lstrip())
        written.append(str(path.relative_to(source_root)))
        if changed:
            path.chmod(0o644)
    return {
        "status": "completed",
        "strategy": strategy,
        "compatibility_basis": "OSCAR README states inference can fall back to PyTorch SDPA without TransformerEngine",
        "files_written": written,
        "raw_secret_values_recorded": False,
    }


def _framework_probe(python: str, source_root: Path | None = None) -> dict[str, Any]:
    code = (
        "import importlib.util, json\n"
        "payload={'torch_importable': False, 'torch_cuda_available': False, "
        "'cuda_device_count': 0, 'transformer_engine_importable': False, "
        "'transformer_engine_blueprint_compat_shim': False, "
        "'pynvml_importable': False, 'loguru_importable': False, "
        "'worldsim_runtime_imports': {}}\n"
        "try:\n"
        " import torch\n"
        " payload['torch_importable']=True\n"
        " payload['torch_version']=getattr(torch, '__version__', None)\n"
        " payload['torch_cuda_available']=bool(torch.cuda.is_available())\n"
        " payload['cuda_device_count']=torch.cuda.device_count()\n"
        "except Exception as exc:\n"
        " payload['torch_error_type']=type(exc).__name__\n"
        "spec = importlib.util.find_spec('transformer_engine')\n"
        "payload['transformer_engine_importable'] = spec is not None\n"
        "payload['transformer_engine_origin'] = getattr(spec, 'origin', None) if spec is not None else None\n"
        "try:\n"
        " import transformer_engine as te\n"
        " payload['transformer_engine_blueprint_compat_shim']=bool(getattr(te, 'BLUEPRINT_COMPAT_SHIM', False))\n"
        "except Exception as exc:\n"
        " payload['transformer_engine_error_type']=type(exc).__name__\n"
        "try:\n"
        " from transformer_engine.common.recipe import DelayedScaling, Format\n"
        " from transformer_engine.pytorch import ops\n"
        " from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast\n"
        " from transformer_engine.pytorch.float8_tensor import Float8Tensor\n"
        " from transformer_engine.pytorch import Linear, LayerNormLinear\n"
        " from transformer_engine.pytorch.tensor import QuantizedTensor\n"
        " payload['transformer_engine_tensor_api_importable']=True\n"
        " payload['transformer_engine_tensor_api_classes']=[getattr(QuantizedTensor, '__name__', 'QuantizedTensor'), getattr(Float8Tensor, '__name__', 'Float8Tensor')]\n"
        " payload['transformer_engine_fp8_api_importable']=True\n"
        " payload['transformer_engine_recipe_api_importable']=True\n"
        " payload['transformer_engine_module_api_importable']=True\n"
        " payload['transformer_engine_ops_api_importable']=True\n"
        " payload['transformer_engine_module_api_classes']=[getattr(Linear, '__name__', 'Linear'), getattr(LayerNormLinear, '__name__', 'LayerNormLinear'), getattr(ops.Sequential, '__name__', 'Sequential')]\n"
        " payload['transformer_engine_fp8_enabled']=bool(FP8GlobalStateManager.is_fp8_enabled())\n"
        " payload['transformer_engine_recipe_format_names']=[getattr(Format.E4M3, '__name__', getattr(Format.E4M3, 'name', 'E4M3')), getattr(Format.HYBRID, '__name__', getattr(Format.HYBRID, 'name', 'HYBRID')), getattr(DelayedScaling, '__name__', 'DelayedScaling')]\n"
        "except Exception as exc:\n"
        " payload['transformer_engine_tensor_api_importable']=False\n"
        " payload['transformer_engine_fp8_api_importable']=False\n"
        " payload['transformer_engine_recipe_api_importable']=False\n"
        " payload['transformer_engine_module_api_importable']=False\n"
        " payload['transformer_engine_ops_api_importable']=False\n"
        " payload['transformer_engine_tensor_api_error_type']=type(exc).__name__\n"
        "pynvml_spec = importlib.util.find_spec('pynvml')\n"
        "payload['pynvml_importable'] = pynvml_spec is not None\n"
        "payload['pynvml_origin'] = getattr(pynvml_spec, 'origin', None) if pynvml_spec is not None else None\n"
        "loguru_spec = importlib.util.find_spec('loguru')\n"
        "payload['loguru_importable'] = loguru_spec is not None\n"
        "payload['loguru_origin'] = getattr(loguru_spec, 'origin', None) if loguru_spec is not None else None\n"
        "worldsim_runtime_modules = {\n"
        " 'attrs':'attrs', 'av':'av', 'boto3':'boto3', 'botocore':'botocore',\n"
        " 'cv2':'cv2', 'decord':'decord', 'fvcore':'fvcore', 'hydra':'hydra',\n"
        " 'matplotlib':'matplotlib', 'megatron_core':'megatron.core',\n"
        " 'omegaconf':'omegaconf', 'onnx':'onnx', 'onnxscript':'onnxscript',\n"
        " 'pandas':'pandas', 'pytest':'pytest', 'qwen_vl_utils':'qwen_vl_utils', 'termcolor':'termcolor',\n"
        " 'torchmetrics':'torchmetrics', 'wandb':'wandb', 'webdataset':'webdataset'}\n"
        "payload['worldsim_runtime_imports'] = {}\n"
        "for label, module in worldsim_runtime_modules.items():\n"
        " module_spec = importlib.util.find_spec(module)\n"
        " payload['worldsim_runtime_imports'][label] = {\n"
        "  'module': module,\n"
        "  'importable': module_spec is not None,\n"
        "  'origin': getattr(module_spec, 'origin', None) if module_spec is not None else None,\n"
        " }\n"
        "print(json.dumps(payload))\n"
    )
    detail = _run([python, "-c", code], timeout=120, env=_python_env_for_source(source_root))
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(detail.get("stdout_tail_redacted") or "{}")
    except Exception:
        payload = {}
    return {"status": "completed", "payload": payload, "subprocess": detail}


def _pip_install_argv(python: str, *args: str) -> list[str]:
    argv = [python, "-m", "pip", "install"]
    allow_break_system = os.environ.get(
        "BLUEPRINT_WAM_PROVIDER_ALLOW_BREAK_SYSTEM_PACKAGES",
        "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if allow_break_system:
        argv.append("--break-system-packages")
    argv.extend(args)
    return argv


def _ensure_dependencies(python: str, source_root: Path) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    transformer_engine_strategy = os.environ.get(
        "BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY",
        "torch_sdpa_compat_shim",
    ).strip() or "torch_sdpa_compat_shim"
    transformer_engine_optional = transformer_engine_strategy in {"none", "disabled"}
    framework_before = _framework_probe(python, source_root)
    framework_before_payload = _mapping(framework_before.get("payload"))
    system_torch_available = framework_before_payload.get("torch_cuda_available") is True
    transformer_engine_available = framework_before_payload.get("transformer_engine_importable") is True
    transformer_engine_is_compat_shim = (
        framework_before_payload.get("transformer_engine_blueprint_compat_shim") is True
    )
    skip_runtime_pip_install = os.environ.get(
        "BLUEPRINT_OSCAR_WAM_SKIP_RUNTIME_PIP_INSTALL",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if skip_runtime_pip_install:
        blockers: list[str] = []
        if not system_torch_available:
            blockers.append("image_runtime_torch_cuda_unavailable")
        if not transformer_engine_optional:
            if not transformer_engine_available:
                blockers.append("image_runtime_transformer_engine_or_shim_unavailable")
            if framework_before_payload.get("transformer_engine_tensor_api_importable") is not True:
                blockers.append("image_runtime_transformer_engine_tensor_api_unavailable")
            if framework_before_payload.get("transformer_engine_fp8_api_importable") is not True:
                blockers.append("image_runtime_transformer_engine_fp8_api_unavailable")
            if framework_before_payload.get("transformer_engine_recipe_api_importable") is not True:
                blockers.append("image_runtime_transformer_engine_recipe_api_unavailable")
            if framework_before_payload.get("transformer_engine_module_api_importable") is not True:
                blockers.append("image_runtime_transformer_engine_module_api_unavailable")
            if framework_before_payload.get("transformer_engine_ops_api_importable") is not True:
                blockers.append("image_runtime_transformer_engine_ops_api_unavailable")
        if framework_before_payload.get("pynvml_importable") is not True:
            blockers.append("image_runtime_pynvml_unavailable")
        if framework_before_payload.get("loguru_importable") is not True:
            blockers.append("image_runtime_loguru_unavailable")
        worldsim_runtime_imports = _mapping(
            framework_before_payload.get("worldsim_runtime_imports")
        )
        for label, detail in sorted(worldsim_runtime_imports.items()):
            if _mapping(detail).get("importable") is not True:
                blockers.append(f"image_runtime_worldsim_extra_unavailable:{label}")
        return {
            "status": "completed" if not blockers else "blocked",
            "source_requirements_file": None,
            "framework_probe_before_install": framework_before,
            "framework_probe_after_requirements": framework_before,
            "framework_probe_after_transformer_engine": framework_before,
            "system_torch_reused": system_torch_available,
            "transformer_engine_available_before_install": transformer_engine_available,
            "transformer_engine_compat_shim_available_before_install": transformer_engine_is_compat_shim,
            "transformer_engine_strategy": transformer_engine_strategy,
            "transformer_engine_optional": transformer_engine_optional,
            "attempted_real_transformer_engine_install": False,
            "runtime_pip_install_skipped_by_reusable_image": True,
            "commands": commands,
            "blockers": blockers,
        }
    base_packages = [
        "huggingface_hub",
        "hf_transfer",
        "opencv-python-headless",
        "imageio",
        "imageio-ffmpeg",
        "ffmpegcv",
        "nvidia-resiliency-ext>=0.6.0",
        "peft",
        "pytest",
    ]
    allow_break_system_packages = os.environ.get(
        "BLUEPRINT_WAM_PROVIDER_ALLOW_BREAK_SYSTEM_PACKAGES",
        "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    commands.append(_run(_pip_install_argv(python, "--upgrade", "pip"), timeout=600))
    commands.append(_run(_pip_install_argv(python, *base_packages), timeout=900))
    req = source_root / "requirements.txt"
    if not req.is_file():
        req = source_root / "requirements_minimal.txt"
    if req.is_file():
        torch_req = source_root / "requirements_torch_cuda128.txt"
        filtered_req = source_root / "requirements_blueprint_without_torch.txt"
        torch_lines: list[str] = []
        filtered_lines: list[str] = []
        for line in req.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            package_name = stripped.split("==", maxsplit=1)[0].split(">=", maxsplit=1)[0]
            if package_name in {"torch", "torchvision"}:
                torch_lines.append(stripped)
            else:
                filtered_lines.append(line)
        torch_req.write_text(
            "\n".join(torch_lines or ["torch", "torchvision"]) + "\n",
            encoding="utf-8",
        )
        if system_torch_available:
            commands.append(
                {
                    "argv_redacted": [python, "-m", "pip", "install", "<torch_requirements_skipped>"],
                    "returncode": 0,
                    "duration_seconds": 0.0,
                    "stdout_size_bytes": 0,
                    "stderr_size_bytes": 0,
                    "stdout_tail_redacted": "skipped because CUDA torch is already importable",
                    "stderr_tail_redacted": "",
                    "raw_secret_values_recorded": False,
                }
            )
        else:
            commands.append(
                _run(
                    _pip_install_argv(
                        python,
                        "--index-url",
                        "https://download.pytorch.org/whl/cu128",
                        "-r",
                        str(torch_req),
                    ),
                    cwd=source_root,
                    timeout=1800,
                )
            )
        filtered_req.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
        commands.append(
            _run(_pip_install_argv(python, "-r", str(filtered_req)), cwd=source_root, timeout=2400)
        )
    framework_after_requirements = _framework_probe(python, source_root)
    framework_after_requirements_payload = _mapping(framework_after_requirements.get("payload"))
    should_attempt_real_te_install = os.environ.get(
        "BLUEPRINT_OSCAR_WAM_ATTEMPT_TRANSFORMER_ENGINE_INSTALL",
        "",
    ).strip().lower() in {"1", "true", "yes"}
    if (
        not transformer_engine_optional
        and (
            framework_after_requirements_payload.get("transformer_engine_importable") is not True
            or (
                framework_after_requirements_payload.get("transformer_engine_blueprint_compat_shim") is True
                and should_attempt_real_te_install
            )
        )
    ):
        te_env = os.environ.copy()
        te_env["NVTE_FRAMEWORK"] = "pytorch"
        commands.append(
            _run(
                _pip_install_argv(
                    python,
                    "--no-build-isolation",
                    "transformer_engine[pytorch]",
                ),
                cwd=source_root,
                timeout=3600,
                env=te_env,
            )
        )
    framework_after_transformer_engine = _framework_probe(python, source_root)
    blockers = [f"dependency_command_failed:{index}" for index, row in enumerate(commands) if row.get("returncode") != 0]
    framework_after_transformer_engine_payload = _mapping(framework_after_transformer_engine.get("payload"))
    if (
        not transformer_engine_optional
        and framework_after_transformer_engine_payload.get("transformer_engine_importable") is not True
    ):
        blockers.append("transformer_engine_or_compat_shim_not_importable_after_dependencies")
    return {
        "status": "completed" if not blockers else "blocked",
        "source_requirements_file": str(req) if req.is_file() else None,
        "framework_probe_before_install": framework_before,
        "framework_probe_after_requirements": framework_after_requirements,
        "framework_probe_after_transformer_engine": framework_after_transformer_engine,
        "system_torch_reused": system_torch_available,
        "transformer_engine_available_before_install": transformer_engine_available,
        "transformer_engine_compat_shim_available_before_install": transformer_engine_is_compat_shim,
        "transformer_engine_strategy": transformer_engine_strategy,
        "transformer_engine_optional": transformer_engine_optional,
        "attempted_real_transformer_engine_install": should_attempt_real_te_install,
        "allow_break_system_packages": allow_break_system_packages,
        "commands": commands,
        "blockers": blockers,
    }


def _checkpoint_resolution_timeout_seconds() -> float:
    raw = os.environ.get("BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS", "1200").strip()
    try:
        return max(60.0, float(raw))
    except ValueError:
        return 1200.0


def _path_inventory(path: Path) -> dict[str, Any]:
    file_count = 0
    total_size_bytes = 0
    largest_file_size_bytes = 0
    if path.exists():
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            try:
                size = item.stat().st_size
            except OSError:
                size = 0
            file_count += 1
            total_size_bytes += size
            largest_file_size_bytes = max(largest_file_size_bytes, size)
    return {
        "file_count": file_count,
        "total_size_bytes": total_size_bytes,
        "largest_file_size_bytes": largest_file_size_bytes,
    }


def _file_tail(path: Path, *, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit), os.SEEK_SET)
            data = handle.read(limit)
    except OSError:
        return ""
    return _redacted_tail(data.decode("utf-8", errors="replace"), limit=limit)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=20)
        return
    except Exception:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        process.kill()
    try:
        process.wait(timeout=20)
    except Exception:
        pass


def _run_checkpoint_download(
    argv: list[str],
    *,
    env: Mapping[str, str],
    target: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    stdout_path = target.parent / "snapshot_download.stdout.log"
    stderr_path = target.parent / "snapshot_download.stderr.log"
    timeout = max(60.0, float(timeout_seconds))
    _phase(
        "checkpoint_download_subprocess_started",
        argv0=Path(argv[0]).name if argv else "",
        target=str(target),
        timeout_seconds=timeout,
    )
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr:
        process = subprocess.Popen(
            argv,
            env=dict(env),
            text=True,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        last_heartbeat = started
        timed_out = False
        while True:
            returncode = process.poll()
            now = time.monotonic()
            elapsed = now - started
            if returncode is not None:
                break
            if elapsed >= timeout:
                timed_out = True
                inventory = _path_inventory(target)
                _phase(
                    "checkpoint_download_timeout_reached",
                    elapsed_seconds=round(elapsed, 3),
                    timeout_seconds=timeout,
                    target=str(target),
                    **inventory,
                )
                _terminate_process_group(process)
                returncode = process.poll()
                break
            if now - last_heartbeat >= 60:
                inventory = _path_inventory(target)
                _phase(
                    "checkpoint_download_waiting",
                    elapsed_seconds=round(elapsed, 3),
                    timeout_seconds=timeout,
                    target=str(target),
                    **inventory,
                )
                last_heartbeat = now
            time.sleep(5)
    duration = round(time.monotonic() - started, 6)
    stdout_size = stdout_path.stat().st_size if stdout_path.is_file() else 0
    stderr_size = stderr_path.stat().st_size if stderr_path.is_file() else 0
    detail = {
        "argv_redacted": [argv[0], "-c", "<huggingface_snapshot_download>"] if argv else [],
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout,
        "duration_seconds": duration,
        "stdout_size_bytes": stdout_size,
        "stderr_size_bytes": stderr_size,
        "stdout_tail_redacted": _file_tail(stdout_path),
        "stderr_tail_redacted": _file_tail(stderr_path),
        "stdout_log_path": str(stdout_path),
        "stderr_log_path": str(stderr_path),
        "checkpoint_inventory": _path_inventory(target),
        "raw_secret_values_recorded": False,
    }
    _phase(
        "checkpoint_download_subprocess_completed",
        returncode=returncode,
        timed_out=timed_out,
        duration_seconds=duration,
        **_mapping(detail.get("checkpoint_inventory")),
    )
    return detail


def _checkpoint(work_dir: Path, python: str, *, timeout_seconds: float) -> tuple[Path | None, dict[str, Any]]:
    configured = os.environ.get("BLUEPRINT_OSCAR_WAM_CHECKPOINT", "").strip()
    if configured and Path(configured).exists():
        return Path(configured).resolve(), {
            "status": "completed",
            "source": "configured_path",
            "path": str(Path(configured).resolve()),
            "resolution_timeout_seconds": timeout_seconds,
        }
    target = work_dir / "checkpoints" / "oscar_2b"
    if target.exists() and any(target.rglob("*")):
        return target, {
            "status": "completed",
            "source": "existing_cache",
            "path": str(target),
            "resolution_timeout_seconds": timeout_seconds,
        }
    code = (
        "from huggingface_hub import snapshot_download\n"
        "import os, sys\n"
        "repo=os.environ['BLUEPRINT_OSCAR_WAM_HF_REPO']\n"
        "target=os.environ['BLUEPRINT_OSCAR_WAM_CHECKPOINT_TARGET']\n"
        "snapshot_download(repo_id=repo, local_dir=target, local_dir_use_symlinks=False, token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))\n"
    )
    env = os.environ.copy()
    env["BLUEPRINT_OSCAR_WAM_HF_REPO"] = OSCAR_HF_REPO
    env["BLUEPRINT_OSCAR_WAM_CHECKPOINT_TARGET"] = str(target)
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if os.environ.get("BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    target.mkdir(parents=True, exist_ok=True)
    detail = _run_checkpoint_download(
        [python, "-c", code],
        env=env,
        target=target,
        timeout_seconds=timeout_seconds,
    )
    if detail.get("timed_out"):
        return None, {
            "status": "blocked",
            "source": "huggingface_snapshot_download",
            "repo_id": OSCAR_HF_REPO,
            "path": str(target),
            "resolution_timeout_seconds": timeout_seconds,
            "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            "raw_hf_token_recorded": False,
            "blockers": ["oscar_checkpoint_download_timeout"],
            "retry_command_redacted": [python, "-c", "<huggingface_snapshot_download>"],
            "retry_env": {
                "BLUEPRINT_OSCAR_WAM_HF_REPO": OSCAR_HF_REPO,
                "BLUEPRINT_OSCAR_WAM_CHECKPOINT_TARGET": str(target),
                "BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER": (
                    "configured"
                    if os.environ.get("BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER")
                    else "default_true"
                ),
                "HF_TOKEN": "configured" if os.environ.get("HF_TOKEN") else "missing",
                "HUGGING_FACE_HUB_TOKEN": "configured" if os.environ.get("HUGGING_FACE_HUB_TOKEN") else "missing",
            },
            "subprocess": detail,
        }
    blockers = []
    if detail.get("returncode") != 0:
        blockers.append("oscar_checkpoint_download_failed")
    if not any(target.rglob("*")):
        blockers.append("oscar_checkpoint_directory_empty_after_download")
    return (target if not blockers else None), {
        "status": "completed" if not blockers else "blocked",
        "source": "huggingface_snapshot_download",
        "repo_id": OSCAR_HF_REPO,
        "path": str(target),
        "resolution_timeout_seconds": timeout_seconds,
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "raw_hf_token_recorded": False,
        "blockers": blockers,
        "retry_command_redacted": [python, "-c", "<huggingface_snapshot_download>"],
            "retry_env": {
                "BLUEPRINT_OSCAR_WAM_HF_REPO": OSCAR_HF_REPO,
                "BLUEPRINT_OSCAR_WAM_CHECKPOINT_TARGET": str(target),
                "BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER": (
                    "configured"
                    if os.environ.get("BLUEPRINT_OSCAR_WAM_ENABLE_HF_TRANSFER")
                    else "default_true"
                ),
                "HF_TOKEN": "configured" if os.environ.get("HF_TOKEN") else "missing",
                "HUGGING_FACE_HUB_TOKEN": "configured" if os.environ.get("HUGGING_FACE_HUB_TOKEN") else "missing",
            },
        "subprocess": detail,
    }


def _cuda_probe(python: str) -> dict[str, Any]:
    code = "import json, torch; print(json.dumps({'torch_cuda_available': bool(torch.cuda.is_available()), 'cuda_device_count': torch.cuda.device_count()}))"
    detail = _run([python, "-c", code], timeout=120)
    payload: dict[str, Any] = {}
    try:
        completed = subprocess.run([python, "-c", code], text=True, capture_output=True, check=False, timeout=120)
        payload = json.loads(completed.stdout or "{}")
        detail = {
            "argv_redacted": [python, "-c", "<torch_cuda_probe>"],
            "returncode": completed.returncode,
            "stdout_size_bytes": len(completed.stdout or ""),
            "stderr_size_bytes": len(completed.stderr or ""),
            "stderr_omitted_to_avoid_secret_leakage": bool(completed.stderr),
        }
    except Exception:
        payload = {}
    blockers = []
    if payload.get("torch_cuda_available") is not True:
        blockers.append("blocked_oscar_requires_cuda_gpu_runtime")
    return {"status": "completed" if not blockers else "blocked", "payload": payload, "blockers": blockers, "subprocess": detail}


def main() -> int:
    started = time.monotonic()
    _phase("runtime_started")
    bundle_dir = Path(os.environ.get("BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR", Path.cwd())).resolve()
    output_dir = Path(os.environ.get("BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR", bundle_dir / "runtime_output")).resolve()
    work_dir = Path(
        os.environ.get("BLUEPRINT_WAM_PROVIDER_WORK_DIR", bundle_dir / "runtime_work")
    ).resolve()
    runtime_manifest_path = bundle_dir / "provider_runtime" / "wam_provider_runtime_manifest.json"
    rollout_input_path = Path(os.environ.get("BLUEPRINT_WAM_ROLLOUT_INPUT", bundle_dir / "provider_runtime" / "wam_rollout_input_manifest.json")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "wam_runtime_result.json"
    provider_output_path = output_dir / "wam_provider_output.json"
    generated_video = output_dir / "oscar_generated_rollout.mp4"
    _phase("python_bootstrap_started")
    python, python_bootstrap = _bootstrap_python(work_dir)
    _phase("python_bootstrap_completed", status=python_bootstrap.get("status"), python=python)
    blockers: list[str] = []
    if python_bootstrap.get("status") != "completed":
        blockers.extend(python_bootstrap.get("blockers") or ["wam_provider_python_bootstrap_failed"])
    runtime_manifest = _mapping(json.loads(runtime_manifest_path.read_text(encoding="utf-8"))) if runtime_manifest_path.is_file() else {}
    rollout_input = _mapping(json.loads(rollout_input_path.read_text(encoding="utf-8"))) if rollout_input_path.is_file() else {}
    _phase(
        "inputs_loaded",
        runtime_manifest_present=bool(runtime_manifest),
        rollout_input_present=bool(rollout_input),
    )
    _phase("source_clone_started")
    source_root, source_detail = _clone_source(work_dir)
    _phase("source_clone_completed", status=source_detail.get("status"), source_present=source_root is not None)
    if source_root is None:
        blockers.extend(source_detail.get("blockers") or ["oscar_source_unavailable"])
    source_compatibility_detail: dict[str, Any] = {"status": "not_run"}
    if source_root is not None and not blockers:
        _phase("source_compatibility_started")
        source_compatibility_detail = _apply_oscar_source_compatibility(source_root)
        _phase(
            "source_compatibility_completed",
            status=source_compatibility_detail.get("status"),
            strategy=source_compatibility_detail.get("strategy"),
        )
        if source_compatibility_detail.get("status") == "blocked":
            blockers.extend(
                source_compatibility_detail.get("blockers")
                or ["oscar_source_compatibility_patch_failed"]
            )
    dependency_detail: dict[str, Any] = {"status": "not_run"}
    checkpoint_path: Path | None = None
    checkpoint_detail: dict[str, Any] = {"status": "not_run"}
    if source_root is not None and not blockers:
        _phase("dependency_setup_started")
        dependency_detail = _ensure_dependencies(python, source_root)
        _phase(
            "dependency_setup_completed",
            status=dependency_detail.get("status"),
            blockers=dependency_detail.get("blockers") or [],
        )
        blockers.extend(dependency_detail.get("blockers") or [])
    cuda: dict[str, Any] = {"status": "not_run"}
    if not blockers:
        _phase("cuda_probe_started")
        cuda = _cuda_probe(python)
        _phase("cuda_probe_completed", status=cuda.get("status"), blockers=cuda.get("blockers") or [])
        if cuda.get("status") != "completed":
            blockers.extend(cuda.get("blockers") or [])
    if not blockers:
        checkpoint_timeout_seconds = _checkpoint_resolution_timeout_seconds()
        _phase(
            "checkpoint_resolution_started",
            source="configured_path_or_existing_cache_or_huggingface_snapshot_download",
            repo_id=OSCAR_HF_REPO,
            timeout_seconds=checkpoint_timeout_seconds,
            configured_checkpoint_path_present=bool(os.environ.get("BLUEPRINT_OSCAR_WAM_CHECKPOINT", "").strip()),
        )
        checkpoint_path, checkpoint_detail = _checkpoint(
            work_dir,
            python,
            timeout_seconds=checkpoint_timeout_seconds,
        )
        _phase(
            "checkpoint_resolution_completed",
            status=checkpoint_detail.get("status"),
            source=checkpoint_detail.get("source"),
            checkpoint_present=checkpoint_path is not None,
            blockers=checkpoint_detail.get("blockers") or [],
        )
        if checkpoint_path is None:
            blockers.extend(checkpoint_detail.get("blockers") or ["oscar_checkpoint_unavailable"])
    inference_detail: dict[str, Any] = {"status": "not_run"}
    if not blockers and source_root is not None and checkpoint_path is not None:
        inference_checkpoint_path = checkpoint_path
        checkpoint_detail["inference_checkpoint_path"] = str(inference_checkpoint_path)
        checkpoint_detail["inference_checkpoint_source"] = "checkpoint_path"
        checkpoint_detail["oscar_loader_appends_model_subdirectory"] = bool(
            checkpoint_path.is_dir() and (checkpoint_path / "model").exists()
        )
        first_frame = bundle_dir / "provider_runtime" / "oscar_input" / "first_frame.png"
        skeleton_video = bundle_dir / "provider_runtime" / "oscar_input" / "blueprint_proxy_skeleton_conditioning.mp4"
        rgb_video = bundle_dir / "provider_runtime" / "oscar_input" / "rgb_context.mp4"
        prompt = runtime_manifest.get("prompt") or "Predict the next robot-scene frames from Blueprint action conditioning."
        start_frame = "0"
        seed = str(runtime_manifest.get("seed") or 42)
        official_case_smoke = str(runtime_manifest.get("official_case_smoke") or "").strip()
        if official_case_smoke:
            checkpoint_roots = [
                checkpoint_path,
                checkpoint_path.parent,
                checkpoint_path.parent.parent,
            ]
            asset_dir = None
            for root in checkpoint_roots:
                candidate = root / "assets" / official_case_smoke
                if (
                    candidate / "rgb.mp4"
                ).is_file() and (candidate / "gripper_scenario.mp4").is_file():
                    asset_dir = candidate
                    break
            case_map = {}
            for root in checkpoint_roots:
                case_map_path = root / "case_map.json"
                if case_map_path.is_file():
                    try:
                        case_map = json.loads(case_map_path.read_text(encoding="utf-8"))
                        break
                    except json.JSONDecodeError:
                        case_map = {}
            if asset_dir is None:
                blockers.append("official_oscar_case_assets_missing")
            else:
                case_detail = _mapping(case_map.get(official_case_smoke))
                start_frame = str(int(case_detail.get("start_frame") or 0))
                official_first_frame = output_dir / f"{official_case_smoke}_first_frame.png"
                try:
                    import cv2

                    capture = cv2.VideoCapture(str(asset_dir / "rgb.mp4"))
                    try:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
                        ok, frame = capture.read()
                    finally:
                        capture.release()
                    if not ok or frame is None:
                        raise RuntimeError("official_case_rgb_frame_decode_failed")
                    cv2.imwrite(str(official_first_frame), frame)
                    caption_path = asset_dir / "caption.pickle"
                    with caption_path.open("rb") as handle:
                        caption = pickle.load(handle)
                    if isinstance(caption, str):
                        prompt = caption
                    elif isinstance(caption, Mapping) and "caption" in caption:
                        prompt = str(caption["caption"])
                    else:
                        prompt = str(caption)
                    first_frame = official_first_frame
                    skeleton_video = asset_dir / "gripper_scenario.mp4"
                    rgb_video = asset_dir / "rgb.mp4"
                    checkpoint_detail["official_case_smoke"] = official_case_smoke
                    checkpoint_detail["official_case_asset_dir"] = str(asset_dir)
                    checkpoint_detail["official_case_start_frame"] = int(start_frame)
                except Exception as exc:
                    blockers.append(f"official_oscar_case_prepare_failed:{type(exc).__name__}")
        argv = [
            python,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "inference/inference_oscar.py",
            "--checkpoint",
            str(inference_checkpoint_path),
            "--first-frame",
            str(first_frame),
            "--skeleton-video",
            str(skeleton_video),
            "--start-frame",
            start_frame,
            "--prompt",
            str(prompt),
            "--num-steps",
            str(runtime_manifest.get("num_steps") or 35),
            "--guidance",
            str(runtime_manifest.get("guidance") or 6.0),
            "--seed",
            seed,
            "--num-frames",
            str(runtime_manifest.get("num_frames") or 81),
            "--height",
            str(runtime_manifest.get("height") or 480),
            "--width",
            str(runtime_manifest.get("width") or 640),
        ]
        omit_fps_arg = bool(
            official_case_smoke
            or os.environ.get("BLUEPRINT_OSCAR_WAM_OMIT_FPS_ARG", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if not omit_fps_arg:
            argv.extend(["--fps", str(runtime_manifest.get("fps") or 15.0)])
        argv.extend(["--output", str(generated_video)])
        runtime_argv_contract = _mapping(runtime_manifest.get("oscar_runtime_argv_contract"))
        runtime_rgb_expected = bool(runtime_argv_contract.get("rgb_video_arg_expected"))
        if rgb_video.is_file() and (official_case_smoke or runtime_rgb_expected):
            argv.extend(["--rgb-video", str(rgb_video)])
        inference_env = os.environ.copy()
        existing_pythonpath = inference_env.get("PYTHONPATH", "")
        inference_env["PYTHONPATH"] = (
            str(source_root)
            if not existing_pythonpath
            else str(source_root) + os.pathsep + existing_pythonpath
        )
        inference_env, cuda_library_env_detail = _prepare_cuda_library_env(work_dir, inference_env)
        _phase(
            "inference_started",
            num_frames=runtime_manifest.get("num_frames") or 81,
            height=runtime_manifest.get("height") or 480,
            width=runtime_manifest.get("width") or 640,
            num_steps=runtime_manifest.get("num_steps") or 35,
            official_case_smoke=official_case_smoke or None,
        )
        if blockers:
            inference_detail = {
                "status": "blocked",
                "returncode": None,
                "duration_seconds": 0.0,
                "argv_redacted": [
                    "<official_case_smoke_preparation_blocked>"
                    if official_case_smoke
                    else "<inference_preparation_blocked>"
                ],
                "stdout_size_bytes": 0,
                "stderr_size_bytes": 0,
                "stderr_omitted_to_avoid_secret_leakage": False,
                "blockers": blockers,
            }
        else:
            inference_detail = _run(
                argv,
                cwd=source_root,
                timeout=int(runtime_manifest.get("timeout_seconds") or 3600),
                env=inference_env,
            )
        _phase(
            "inference_completed",
            returncode=inference_detail.get("returncode"),
            generated_video_present=generated_video.is_file(),
        )
        inference_detail["cuda_library_env"] = cuda_library_env_detail
        inference_detail["argv_redacted"] = [
            "<checkpoint_path_configured>" if item == str(inference_checkpoint_path) else item
            for item in inference_detail["argv_redacted"]
        ]
        if inference_detail.get("returncode") != 0:
            blockers.append("oscar_inference_command_nonzero")
        if not generated_video.is_file():
            discovered = sorted(output_dir.rglob("*.mp4"))
            if discovered:
                shutil.copy2(discovered[0], generated_video)
        if not generated_video.is_file():
            blockers.append("blocked_no_generated_oscar_mp4")
    generated_video_validation = _validate_generated_video(generated_video)
    if generated_video.is_file() and generated_video_validation.get("status") != "completed":
        blockers.append("blocked_generated_oscar_mp4_not_reviewable")
        blockers.extend(generated_video_validation.get("blockers") or [])
    rollouts = []
    if generated_video_validation.get("status") == "completed" and not blockers:
        rollouts.append(
            {
                "rollout_id": "oscar_wam_rollout_0001",
                "policy_id": "oscar_wam_provider_runtime",
                "model_candidate": "oscar_wam",
                "generated_video_path": str(generated_video),
                "model_rollout_confidence": None,
                "generated_rollout_termination_reason": "oscar_inference_command_completed",
                "success_label_source": "generated_video_requires_review",
                "generated_video_review_validation": generated_video_validation,
            }
        )
    status = "completed" if rollouts and not blockers else "blocked"
    provider_payload = {
        "schema_version": "oscar_wam_command_adapter.v1",
        "status": status,
        "adapter_id": "oscar_wam_provider_runtime",
        "rollouts": rollouts,
        "generated_video_count": len(rollouts),
        "model_provenance": {
            "candidate": "oscar_wam",
            "source_url": OSCAR_SOURCE_URL,
            "checkpoint_repo": OSCAR_HF_REPO,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        },
        "input_package": runtime_manifest.get("input_package"),
        "generated_video_review_validation": generated_video_validation,
        "blockers": blockers,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    _write_json(provider_output_path, provider_payload)
    _phase("provider_output_written", status=status, rollout_count=len(rollouts), blockers=blockers)
    runtime_result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "runtime": "oscar_wam_provider_runtime",
        "provider": "vast_or_compatible_cuda",
        "model_candidate": "oscar_wam",
        "model_name": "OSCAR-2B",
        "action_conditioned_video_rollout_generated": bool(rollouts),
        "learned_wam_model_ran": bool(rollouts),
        "generated_video_path": str(generated_video) if generated_video.is_file() else None,
        "generated_video_review_validation": generated_video_validation,
        "rollout_input_manifest_path": str(rollout_input_path),
        "rollout_input_loaded": bool(rollout_input),
        "source_detail": source_detail,
        "source_compatibility_detail": source_compatibility_detail,
        "python_bootstrap": python_bootstrap,
        "dependency_detail": dependency_detail,
        "checkpoint_detail": checkpoint_detail,
        "cuda_probe": cuda,
        "inference_detail": inference_detail,
        "duration_seconds": round(time.monotonic() - started, 6),
        "blockers": blockers,
        "truth_boundary": {
            "generated_video_is_model_output": bool(rollouts),
            "wam_success_label_from_generated_video": False,
            "forward_inverse_consistency_proven": None,
            "forward_inverse_consistency_scored_by_provider_runtime": False,
            "forward_inverse_consistency_requires_external_episode_scorer": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    _write_json(result_path, runtime_result)
    _phase("runtime_result_written", status=status, duration_seconds=round(time.monotonic() - started, 6))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


REMOTE_ENTRYPOINT = r'''#!/usr/bin/env bash
set +e
write_missing_result() {
  local runner_rc="${1:-999}"
  local runner_log="${2:-}"
  local output_dir="${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}"
  mkdir -p "$output_dir"
  python - "$runner_rc" "$runner_log" "$output_dir/wam_runtime_result.json" <<'PY'
import json
import os
import sys
from pathlib import Path

runner_rc = int(sys.argv[1]) if sys.argv[1].isdigit() else 999
runner_log = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
result_path = Path(sys.argv[3])
tail = ""
if runner_log and runner_log.is_file():
    tail = runner_log.read_text(encoding="utf-8", errors="replace")[-4000:]
for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    value = os.environ.get(key)
    if value:
        tail = tail.replace(value, "<redacted-secret>")
payload = {
    "schema_version": "wam_runtime_result.v1",
    "status": "blocked",
    "runtime": "oscar_wam_provider_runtime",
    "blockers": [
        "wam_runner_process_exited_without_runtime_result",
        "blocked_wam_process_exited_without_result",
    ],
    "runner_returncode": runner_rc,
    "runner_log_tail_redacted": tail,
    "action_conditioned_video_rollout_generated": False,
    "learned_wam_model_ran": False,
    "raw_credentials_written_to_artifacts": False,
    "secret_hashes_written_to_artifacts": False,
}
result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}
PYTHON_BIN="${BLUEPRINT_WAM_PROVIDER_PYTHON:-python}"
mkdir -p "${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}"
RUNNER_LOG="${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}/wam_provider_runner.log"
"$PYTHON_BIN" "$(dirname "$0")/wam_provider_runtime_runner.py" 2>&1 | tee "$RUNNER_LOG"
rc=${PIPESTATUS[0]}
if [ ! -f "${BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR:-runtime_output}/wam_runtime_result.json" ]; then
  write_missing_result "$rc" "$RUNNER_LOG"
fi
exit $rc
'''


def _write_runtime_files(
    *,
    runtime_dir: Path,
    rollout_manifest: Mapping[str, Any],
    input_package: Mapping[str, Any],
    oscar_source_url: str,
    oscar_hf_repo: str,
    timeout_seconds: int,
    num_steps: int,
    guidance: float,
    seed: int,
) -> None:
    ensure_dir(runtime_dir)
    oscar_input_dir = runtime_dir / "oscar_input"
    ensure_dir(oscar_input_dir)
    first_frame = Path(_string(_mapping(input_package.get("first_frame")).get("path"))).expanduser()
    skeleton = Path(_string(_mapping(input_package.get("skeleton_video")).get("path"))).expanduser()
    runtime_first_frame = oscar_input_dir / "first_frame.png"
    runtime_skeleton = oscar_input_dir / "blueprint_proxy_skeleton_conditioning.mp4"
    rgb_package = _mapping(input_package.get("rgb_video"))
    rgb_context_mode = _string(rgb_package.get("rgb_context_mode")) or "auto"
    projected_conditioning_used = _package_uses_projected_g1_skeleton(input_package)
    rgb_context_requested = bool(
        rgb_package.get("used_for_oscar_rgb_latent_context") is not False
        if not projected_conditioning_used
        else rgb_package.get("used_for_oscar_rgb_latent_context") is True
    )
    rgb_source = Path(_string(rgb_package.get("path"))).expanduser()
    if not rgb_source.is_file():
        rgb_source = Path(_string(input_package.get("source_review_video_path"))).expanduser()
    runtime_rgb = oscar_input_dir / "rgb_context.mp4"
    projected_trace_source = Path(
        _string(_mapping(input_package.get("projected_skeleton_trace")).get("path"))
    ).expanduser()
    runtime_projected_trace = oscar_input_dir / "g1_projected_skeleton_trace.jsonl"
    _copy_file(first_frame, runtime_first_frame)
    _copy_file(skeleton, runtime_skeleton)
    rgb_runtime_path = None
    if rgb_context_requested and rgb_source.is_file():
        _copy_file(rgb_source, runtime_rgb)
        rgb_runtime_path = "provider_runtime/oscar_input/rgb_context.mp4"
    projected_skeleton_runtime_path = None
    if projected_trace_source.is_file():
        _copy_file(projected_trace_source, runtime_projected_trace)
        projected_skeleton_runtime_path = (
            "provider_runtime/oscar_input/g1_projected_skeleton_trace.jsonl"
        )
    runtime_input_package = _runtime_input_package_manifest(
        input_package,
        first_frame_runtime_path="provider_runtime/oscar_input/first_frame.png",
        skeleton_runtime_path=(
            "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4"
        ),
        rgb_runtime_path=rgb_runtime_path,
        projected_skeleton_runtime_path=projected_skeleton_runtime_path,
    )
    write_json(
        runtime_dir / "wam_rollout_input_manifest.json",
        _runtime_rollout_manifest(
            rollout_manifest,
            projected_skeleton_runtime_path=projected_skeleton_runtime_path,
        ),
    )
    runtime_manifest = {
        "schema_version": "wam_provider_runtime_manifest.v1",
        "runtime": "oscar_wam_provider_runtime",
        "model_candidate": "oscar_wam",
        "model_name": "OSCAR-2B",
        "oscar_source_url": oscar_source_url,
        "oscar_hf_repo": oscar_hf_repo,
        "prompt": input_package.get("prompt"),
        "input_package": runtime_input_package,
        "num_frames": input_package.get("num_frames") or DEFAULT_NUM_FRAMES,
        "fps": input_package.get("fps") or DEFAULT_FPS,
        "height": input_package.get("height") or DEFAULT_HEIGHT,
        "width": input_package.get("width") or DEFAULT_WIDTH,
        "num_steps": num_steps,
        "guidance": guidance,
        "seed": seed,
        "official_case_smoke": os.environ.get(
            "BLUEPRINT_OSCAR_WAM_OFFICIAL_CASE_SMOKE", ""
        ).strip(),
        "timeout_seconds": timeout_seconds,
        "remote_secret_contract": {
            "hf_token_env_supported": ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
            "raw_tokens_written_to_artifacts": False,
            "token_hashes_written_to_artifacts": False,
        },
        "oscar_runtime_argv_contract": {
            "first_frame_arg": "provider_runtime/oscar_input/first_frame.png",
            "skeleton_video_arg": (
                "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4"
            ),
            "rgb_context_packaged": bool(rgb_runtime_path),
            "rgb_context_mode": rgb_context_mode,
            "rgb_video_arg_expected": bool(rgb_runtime_path),
            "rgb_video_arg": "provider_runtime/oscar_input/rgb_context.mp4"
            if rgb_runtime_path
            else None,
            "rgb_video_arg_omitted_by_rgb_context_mode": bool(
                rgb_context_mode == "never" and not rgb_runtime_path
            ),
            "projected_g1_skeleton_conditioning_suppresses_rgb_context": bool(
                projected_conditioning_used and not rgb_runtime_path
            ),
            "projected_g1_rgb_context_enabled": bool(
                projected_conditioning_used and rgb_runtime_path
            ),
            "projected_skeleton_trace_packaged": bool(projected_skeleton_runtime_path),
            "projected_skeleton_trace_runtime_path": projected_skeleton_runtime_path,
            "remote_runner_records_actual_argv_redacted_in_wam_runtime_result": True,
            "raw_secret_values_recorded": False,
        },
        "truth_boundary": {
            "model_backend_replaceable": True,
            "generated_rollout_not_physical_robot_proof": True,
            "generated_success_label_requires_external_vlm_or_human_judge": True,
            "rgb_context_packaging_is_not_visual_usefulness_proof": True,
            "projected_g1_skeleton_packaging_is_not_visual_usefulness_proof": True,
        },
    }
    write_json(runtime_dir / "wam_provider_runtime_manifest.json", runtime_manifest)
    runner = runtime_dir / "wam_provider_runtime_runner.py"
    runner.write_text(REMOTE_RUNNER, encoding="utf-8")
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    entrypoint = runtime_dir / "run_wam_provider_runtime.sh"
    entrypoint.write_text(REMOTE_ENTRYPOINT, encoding="utf-8")
    entrypoint.chmod(entrypoint.stat().st_mode | stat.S_IXUSR)


def build_oscar_wam_provider_bundle(
    *,
    job_dir: str | Path,
    wam_rollout_input_manifest: str | Path,
    oscar_input_dir: str | Path | None = None,
    oscar_input_package_manifest: str | Path | None = None,
    oscar_source_url: str = DEFAULT_OSCAR_SOURCE_URL,
    oscar_hf_repo: str = DEFAULT_OSCAR_HF_REPO,
    timeout_seconds: int = 3600,
    num_steps: int = 35,
    guidance: float = 6.0,
    seed: int = 42,
    num_frames: int = DEFAULT_NUM_FRAMES,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    fps: float = DEFAULT_FPS,
    bundle_filename: str = DEFAULT_BUNDLE_FILENAME,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_rollout_input = Path(wam_rollout_input_manifest).expanduser().resolve()
    ensure_dir(resolved_job_dir)
    bundle_root = resolved_job_dir / "oscar_wam_provider_bundle"
    runtime_dir = bundle_root / "provider_runtime"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    ensure_dir(runtime_dir)
    blockers: list[str] = []
    rollout_manifest: dict[str, Any] = {}
    input_package: dict[str, Any] = {}
    if not resolved_rollout_input.is_file():
        blockers.append("wam_rollout_input_manifest_missing")
    else:
        rollout_manifest = _read_json(resolved_rollout_input)
    materialization_error: dict[str, Any] = {}
    try:
        if not blockers and oscar_input_dir:
            resolved_input_dir = Path(oscar_input_dir).expanduser().resolve()
            resolved_package_manifest = (
                Path(oscar_input_package_manifest).expanduser().resolve()
                if oscar_input_package_manifest
                else None
            )
            input_package = _materialized_package_from_existing(
                oscar_input_dir=resolved_input_dir,
                package_manifest_path=resolved_package_manifest,
                rollout_manifest=rollout_manifest,
            )
        elif not blockers and _string(rollout_manifest.get("schema_version")) == (
            "wam_generation_step_input.v1"
        ):
            workspace = resolved_job_dir / "local_input_materialization"
            input_package = _materialize_oscar_input_package_from_wam_generation_step(
                step_input=rollout_manifest,
                work_dir=workspace,
                width=width,
                height=height,
                fps=fps,
                num_frames=num_frames,
            )
        elif not blockers:
            workspace = resolved_job_dir / "local_input_materialization"
            input_package = _materialize_oscar_input_package(
                rollout_manifest=rollout_manifest,
                work_dir=workspace,
                width=width,
                height=height,
                fps=fps,
                num_frames=num_frames,
            )
    except Exception as exc:
        materialization_error = {
            "type": type(exc).__name__,
            "message": _safe_error_text(exc),
            "raw_message_omitted_if_path_like": bool("/" in str(exc) or "\\" in str(exc)),
        }
        blockers.append(f"oscar_wam_input_package_materialization_failed:{type(exc).__name__}")
        if materialization_error["message"]:
            blockers.append(
                "oscar_wam_input_package_materialization_error:"
                + str(materialization_error["message"])
            )
    conditioning_video_blockers: list[str] = []
    if not blockers:
        conditioning_video_blockers = _conditioning_video_input_blockers(input_package)
        blockers.extend(conditioning_video_blockers)
    if not blockers:
        _write_runtime_files(
            runtime_dir=runtime_dir,
            rollout_manifest=rollout_manifest,
            input_package=input_package,
            oscar_source_url=oscar_source_url,
            oscar_hf_repo=oscar_hf_repo,
            timeout_seconds=timeout_seconds,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
    bundle_path = resolved_job_dir / bundle_filename
    zip_entries: list[str] = []
    if not blockers:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle_root).as_posix())
        with zipfile.ZipFile(bundle_path) as archive:
            zip_entries = sorted(archive.namelist())
            if archive.testzip() is not None:
                blockers.append("provider_runtime_bundle_zip_integrity_failed")
    manifest = {
        "schema_version": OSCAR_WAM_PROVIDER_BUNDLE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed" if not blockers else "blocked",
        "job_dir": str(resolved_job_dir),
        "bundle_path": str(bundle_path),
        "bundle_present": bundle_path.is_file(),
        "bundle_size_bytes": bundle_path.stat().st_size if bundle_path.is_file() else 0,
        "local_bundle_ready_for_remote_staging": not blockers,
        "wam_rollout_input_manifest": str(resolved_rollout_input),
        "provider_bundle_kind": "wam",
        "runtime_dir": str(runtime_dir),
        "zip_entry_count": len(zip_entries),
        "zip_entries": zip_entries,
        "oscar_source_url": oscar_source_url,
        "oscar_hf_repo": oscar_hf_repo,
        "input_package_conditioning_video_blockers": conditioning_video_blockers,
        "input_package_materialization_error": materialization_error,
        "input_package_source_schema_version": rollout_manifest.get("schema_version"),
        "blockers": blockers,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "truth_boundary": {
            "bundle_build_is_not_model_execution": True,
            "provider_runtime_must_generate_mp4_before_wam_model_ran_true": True,
        },
    }
    write_json(resolved_job_dir / "oscar_wam_provider_bundle_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--wam-rollout-input-manifest", required=True)
    parser.add_argument("--oscar-input-dir")
    parser.add_argument("--oscar-input-package-manifest")
    parser.add_argument("--oscar-source-url", default=DEFAULT_OSCAR_SOURCE_URL)
    parser.add_argument("--oscar-hf-repo", default=DEFAULT_OSCAR_HF_REPO)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--bundle-filename", default=DEFAULT_BUNDLE_FILENAME)
    args = parser.parse_args(argv)
    manifest = build_oscar_wam_provider_bundle(
        job_dir=args.job_dir,
        wam_rollout_input_manifest=args.wam_rollout_input_manifest,
        oscar_input_dir=args.oscar_input_dir,
        oscar_input_package_manifest=args.oscar_input_package_manifest,
        oscar_source_url=args.oscar_source_url,
        oscar_hf_repo=args.oscar_hf_repo,
        timeout_seconds=args.timeout_seconds,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        bundle_filename=args.bundle_filename,
    )
    print(f"[oscar-wam-provider-bundle] manifest={Path(args.job_dir).resolve() / 'oscar_wam_provider_bundle_manifest.json'}")
    print(f"[oscar-wam-provider-bundle] status={manifest.get('status')}")
    blockers = manifest.get("blockers") or []
    if blockers:
        print("[oscar-wam-provider-bundle] blockers=" + ",".join(str(item) for item in blockers))
    return 0 if manifest.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
