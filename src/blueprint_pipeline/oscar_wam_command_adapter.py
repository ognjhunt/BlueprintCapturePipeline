"""Command adapter for OSCAR action-conditioned WAM rollout generation.

The adapter reads the Blueprint WAM rollout manifest path from
``BLUEPRINT_WAM_ROLLOUT_INPUT``, builds OSCAR's required first-frame plus
conditioning inputs from MuJoCo review/trace artifacts, runs the
public OSCAR inference entrypoint, and writes Blueprint rollout JSON only when
OSCAR produces a generated MP4.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import sha256_file, utc_now_iso
from .oscar_official_release import (
    OFFICIAL_OSCAR_HF_REPO,
    official_release_blockers,
    official_release_contract,
)
from .wam_generated_video_review import (
    validate_generated_mp4_for_review,
    visual_smoke_generated_rollouts_for_review,
)


ADAPTER_ID = "blueprint_oscar_wam_command_adapter"
SCHEMA_VERSION = "oscar_wam_command_adapter.v1"
DEFAULT_NUM_FRAMES = 81
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640
DEFAULT_FPS = 15.0
OSCAR_PUBLIC_SOURCE_REVISION = "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb"
OSCAR_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with "
    "no motion, motion blur, over-saturation, shaky footage, low resolution, "
    "grainy texture, pixelated images, poorly lit areas, underexposed and "
    "overexposed scenes, poor color balance, washed out colors, choppy "
    "sequences, jerky movements, low frame rate, artifacting, color banding, "
    "unnatural transitions, outdated special effects, fake elements, "
    "unconvincing visuals, poorly edited content, jump cuts, visual noise, "
    "and flickering. Overall, the video is of poor quality."
)
ALLOW_EXPERIMENTAL_OSCAR_VERSION_ENV = "BLUEPRINT_ALLOW_EXPERIMENTAL_OSCAR_WAM_VERSION"
DEFAULT_CONDITIONING_BACKGROUND_ALPHA = 0.88
DEFAULT_CONDITIONING_NEAR_BLACK_THRESHOLD = 10
DEFAULT_CONDITIONING_VOID_FILL_BGR = (52, 56, 58)
DEFAULT_CONDITIONING_MODE = "oscar_gripper_scenario_proxy"
FIRST_PERSON_CONDITIONING_MODES = {
    "first_person_review_video",
    "selected_review_video_passthrough",
    "egocentric_review_video_passthrough",
}
EGOCENTRIC_ARM_SKELETON_MODES = {
    "egocentric_arm_skeleton",
    "egocentric_hand_skeleton",
    "first_person_arm_skeleton",
}
TEXTURE_FREE_EGOCENTRIC_ARM_SKELETON_MODES = {
    "texture_free_egocentric_arm_skeleton",
    "oscar_texture_free_egocentric_arm_skeleton",
}
OSCAR_GRIPPER_SCENARIO_PROXY_MODES = {
    "oscar_gripper_scenario_proxy",
    "oscar_egocentric_gripper_proxy",
    "egocentric_rgb_gripper_proxy",
}
PROJECTED_G1_SKELETON_CONDITIONING_MODES = {
    "projected_g1_skeleton",
    "g1_projected_skeleton",
    "unitree_g1_projected_skeleton",
    "projected_g1_arm_hand_skeleton",
}
PROJECTED_G1_SKELETON_RGB_OVERLAY_MODES = {
    "projected_g1_skeleton_rgb_overlay",
    "projected_g1_skeleton_scene_overlay",
    "unitree_g1_projected_skeleton_rgb_overlay",
}
ALL_PROJECTED_G1_SKELETON_MODES = (
    PROJECTED_G1_SKELETON_CONDITIONING_MODES
    | PROJECTED_G1_SKELETON_RGB_OVERLAY_MODES
)
PROXY_SKELETON_CONDITIONING_MODES = {
    "scene_overlay_proxy_skeleton",
    "proxy_skeleton",
    "blueprint_proxy_skeleton",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _rgb_context_mode() -> str:
    raw = _string(os.getenv("BLUEPRINT_OSCAR_WAM_RGB_CONTEXT_MODE", "auto")).lower()
    aliases = {
        "": "auto",
        "default": "auto",
        "on": "always",
        "true": "always",
        "1": "always",
        "yes": "always",
        "off": "never",
        "false": "never",
        "0": "never",
        "no": "never",
        "omit": "never",
        "disabled": "never",
        "disable": "never",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in {"auto", "always", "never"} else "auto"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_src_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _first_existing_path(paths: Sequence[str]) -> Path | None:
    for value in paths:
        if not value:
            continue
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
    return None


def _source_root_from_env() -> Path | None:
    return _first_existing_path(
        [
            os.getenv("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", ""),
            os.getenv("BLUEPRINT_OSCAR_SOURCE_ROOT", ""),
        ]
    )


def _checkpoint_from_env() -> Path | None:
    return _first_existing_path(
        [
            os.getenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", ""),
            os.getenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", ""),
        ]
    )


def _source_root_commit(source_root: Path | None) -> str | None:
    if source_root is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    commit = (completed.stdout or "").strip().splitlines()
    return commit[-1] if commit else None


def _source_root_origin_url(source_root: Path | None) -> str | None:
    if source_root is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "config", "--get", "remote.origin.url"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    urls = [(line or "").strip() for line in (completed.stdout or "").splitlines()]
    return next((url for url in urls if url), None)


def _checkpoint_revision_from_path(checkpoint: Path | None) -> str | None:
    if checkpoint is None:
        return None
    hex_chars = set("0123456789abcdef")
    for item in (checkpoint, *checkpoint.parents):
        name = item.name.strip().lower()
        if len(name) == 40 and all(char in hex_chars for char in name):
            return name
    return None


def _preference_list(env_name: str, default: Sequence[str]) -> list[str]:
    configured = os.getenv(env_name, "")
    values = configured.split(",") if configured else list(default)
    return [_string(value) for value in values if _string(value)]


def _video_candidates(rollout_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in rollout_manifest.get("wam_input_videos", []) or []:
        if isinstance(row, Mapping):
            rows.append({**dict(row), "wam_input_video": True})
    for row in rollout_manifest.get("selected_review_videos", []) or []:
        if isinstance(row, Mapping):
            rows.append(dict(row))
    inputs = _mapping(rollout_manifest.get("inputs"))
    selection_manifest_path = Path(
        _string(inputs.get("review_video_selection_manifest"))
    ).expanduser()
    if selection_manifest_path.is_file():
        selection_manifest = _read_json(selection_manifest_path)
        for row in selection_manifest.get("selected_review_videos", []) or []:
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def _task_prompt_by_run_id(rollout_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in rollout_manifest.get("task_prompts", []) or []:
        if isinstance(row, Mapping) and _string(row.get("scenario_eval_run_id")):
            rows[_string(row.get("scenario_eval_run_id"))] = dict(row)
    return rows


def _selected_video_row(rollout_manifest: Mapping[str, Any]) -> dict[str, Any]:
    camera_preferences = _preference_list(
        "BLUEPRINT_WAM_PREFERRED_CAMERA",
        ("head_pov", "torso_pov", "robot_pov", "robot_follow", "third_person", "overhead"),
    )
    task_preferences = _preference_list(
        "BLUEPRINT_WAM_PREFERRED_TASK_ID",
        ("contact_or_push_light_object", "approach_target"),
    )
    prompt_rows = _task_prompt_by_run_id(rollout_manifest)
    ranked: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for index, candidate in enumerate(_video_candidates(rollout_manifest)):
        path = Path(_string(candidate.get("path"))).expanduser()
        if not path.is_file():
            continue
        row = dict(candidate)
        run_id = _string(row.get("scenario_eval_run_id"))
        if run_id and run_id in prompt_rows:
            row = {**prompt_rows[run_id], **row}
        text = " ".join(
            _string(row.get(key))
            for key in ("task_id", "episode_id", "scenario_eval_run_id", "path")
        )
        camera = _string(row.get("camera"))
        camera_rank = camera_preferences.index(camera) if camera in camera_preferences else len(camera_preferences)
        task_rank = len(task_preferences)
        for pref_index, task_id in enumerate(task_preferences):
            if task_id and task_id in text:
                task_rank = pref_index
                break
        ranked.append(((task_rank, camera_rank, index), {**row, "path": str(path.resolve())}))
    if ranked:
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]
    raise FileNotFoundError("missing_selected_review_video")


def _selected_video_path(rollout_manifest: Mapping[str, Any]) -> Path:
    return Path(_selected_video_row(rollout_manifest)["path"]).resolve()


def _task_prompt(rollout_manifest: Mapping[str, Any]) -> str:
    try:
        selected = _selected_video_row(rollout_manifest)
        prompt = _string(selected.get("task_prompt"))
        if prompt:
            return prompt
    except FileNotFoundError:
        pass
    for row in rollout_manifest.get("task_prompts", []) or []:
        prompt = _string(_mapping(row).get("task_prompt"))
        if prompt:
            return prompt
    raise ValueError("oscar_task_specific_prompt_required")


def _trace_rows(rollout_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = _mapping(rollout_manifest.get("inputs"))
    trace_path = Path(_string(inputs.get("g1_mujoco_locomotion_trace_jsonl"))).expanduser()
    rows = _read_jsonl(trace_path)
    selected_episode = ""
    try:
        selected_video = _selected_video_path(rollout_manifest)
        selected_episode = selected_video.name.split("__", maxsplit=1)[0]
    except FileNotFoundError:
        selected_episode = ""
    if selected_episode:
        episode_rows = [row for row in rows if _string(row.get("episode_id")) == selected_episode]
        if episode_rows:
            return episode_rows
    return rows


def _projected_skeleton_rows(rollout_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = _mapping(rollout_manifest.get("inputs"))
    trace_path = Path(_string(inputs.get("g1_projected_skeleton_trace_jsonl"))).expanduser()
    rows = _read_jsonl(trace_path)
    selected_episode = ""
    try:
        selected_video = _selected_video_path(rollout_manifest)
        selected_episode = selected_video.name.split("__", maxsplit=1)[0]
    except FileNotFoundError:
        selected_episode = ""
    if selected_episode:
        episode_rows = [row for row in rows if _string(row.get("episode_id")) == selected_episode]
        if episode_rows:
            return episode_rows
    return rows


def _projected_skeleton_projectable_row_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if int(row.get("projected_landmark_count") or 0) > 0)


def _package_uses_projected_g1_skeleton(package_manifest: Mapping[str, Any]) -> bool:
    skeleton_video = _mapping(package_manifest.get("skeleton_video"))
    projected_trace = _mapping(package_manifest.get("projected_skeleton_trace"))
    claim_boundary = _mapping(package_manifest.get("claim_boundary"))
    return bool(
        skeleton_video.get("projected_g1_skeleton_rendered")
        or projected_trace.get("used_for_conditioning")
        or claim_boundary.get("projected_g1_skeleton_conditioning_used")
    )


def _configured_conditioning_mode(projected_skeleton_rows: Sequence[Mapping[str, Any]]) -> str:
    configured = os.getenv("BLUEPRINT_OSCAR_WAM_CONDITIONING_MODE")
    if configured is not None and configured.strip():
        return configured.strip()
    if _projected_skeleton_projectable_row_count(projected_skeleton_rows) > 0:
        return "projected_g1_skeleton"
    return DEFAULT_CONDITIONING_MODE


def _conditioning_video_model_input_useful(
    *,
    skeleton_video: Mapping[str, Any],
    visual_smoke: Mapping[str, Any],
) -> bool:
    visual_signal = _mapping(skeleton_video.get("visual_signal"))
    if (
        skeleton_video.get("projected_g1_skeleton_rendered")
        and skeleton_video.get("skeleton_stream_separate_from_rgb")
        and int(skeleton_video.get("projected_g1_skeleton_landmark_draw_count") or 0) > 0
        and visual_signal.get("status") == "completed"
    ):
        return True
    return bool(
        _mapping(visual_smoke.get("claim_boundary")).get(
            "visual_rollout_useful_for_task_success_review"
        )
    )


def _sample_rows(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    if len(rows) == 1:
        return [dict(rows[0]) for _ in range(count)]
    sampled: list[dict[str, Any]] = []
    last = len(rows) - 1
    for index in range(count):
        source_index = round(index * last / max(count - 1, 1))
        sampled.append(dict(rows[source_index]))
    return sampled


def _point_from_root(row: Mapping[str, Any]) -> tuple[float, float, float]:
    position = row.get("root_position")
    if isinstance(position, Sequence) and not isinstance(position, (str, bytes)) and len(position) >= 3:
        return _number(position[0]), _number(position[1]), _number(position[2])
    return 0.0, 0.0, 0.8


def _screen_transform(rows: Sequence[Mapping[str, Any]], width: int, height: int) -> tuple[float, float, float, float]:
    points = [_point_from_root(row) for row in rows] or [(0.0, 0.0, 0.8)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 0.5)
    scale = min(width, height) * 0.46 / span
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    return center_x, center_y, scale, span


def _conditioning_background_settings() -> dict[str, Any]:
    alpha = _number(
        os.getenv("BLUEPRINT_OSCAR_WAM_CONDITIONING_BACKGROUND_ALPHA"),
        DEFAULT_CONDITIONING_BACKGROUND_ALPHA,
    )
    alpha = max(0.0, min(1.0, alpha))
    threshold = int(
        max(
            0,
            min(
                255,
                round(
                    _number(
                        os.getenv("BLUEPRINT_OSCAR_WAM_CONDITIONING_VOID_THRESHOLD"),
                        DEFAULT_CONDITIONING_NEAR_BLACK_THRESHOLD,
                    )
                ),
            ),
        )
    )
    return {
        "background_alpha": alpha,
        "fill_near_black_void": _env_flag(
            "BLUEPRINT_OSCAR_WAM_FILL_NEAR_BLACK_VOID", default=True
        ),
        "near_black_threshold": threshold,
        "void_fill_bgr": list(DEFAULT_CONDITIONING_VOID_FILL_BGR),
        "void_fill_style": _string(
            os.getenv("BLUEPRINT_OSCAR_WAM_VOID_FILL_STYLE", "lab_wall")
        )
        or "lab_wall",
    }


def _fill_near_black_void(
    frame: Any,
    *,
    cv2: Any,
    np: Any,
    settings: Mapping[str, Any],
) -> tuple[Any, int, int]:
    if not bool(settings.get("fill_near_black_void")):
        return frame, 0, 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray <= int(settings.get("near_black_threshold") or 0)
    filled_pixel_count = int(np.count_nonzero(mask))
    total_pixel_count = int(frame.shape[0] * frame.shape[1])
    if filled_pixel_count <= 0:
        return frame, 0, total_pixel_count
    filled = frame.copy()
    style = _string(settings.get("void_fill_style")) or "solid"
    if style == "lab_wall":
        yy, xx = np.indices((frame.shape[0], frame.shape[1]))
        wall = np.zeros_like(frame)
        wall[:, :, 0] = np.clip(70 + yy * 22 // max(frame.shape[0], 1), 0, 255)
        wall[:, :, 1] = np.clip(76 + yy * 18 // max(frame.shape[0], 1), 0, 255)
        wall[:, :, 2] = np.clip(82 + xx * 12 // max(frame.shape[1], 1), 0, 255)
        grid = ((xx % 96) < 2) | ((yy % 96) < 2)
        wall[grid] = np.clip(wall[grid].astype(np.int16) + 18, 0, 255).astype(np.uint8)
        filled[mask] = wall[mask]
    else:
        filled[mask] = np.array(settings.get("void_fill_bgr"), dtype=np.uint8)
    return filled, filled_pixel_count, total_pixel_count


def _render_proxy_skeleton_video(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
    background_video: Path | None = None,
    conditioning_mode: str = "scene_overlay_proxy_skeleton",
    projected_skeleton_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    import cv2
    import math
    import numpy as np

    sampled_rows = _sample_rows(trace_rows, num_frames)
    if not sampled_rows:
        raise ValueError("missing_locomotion_trace_for_oscar_skeleton_conditioning")
    center_x, center_y, scale, _span = _screen_transform(sampled_rows, width, height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("cv2_video_writer_failed_for_oscar_skeleton_conditioning")

    background_frames: list[Any] = []
    background_frame_count = 0
    background_void_fill_pixel_count = 0
    background_total_pixel_count = 0
    background_settings = _conditioning_background_settings()
    first_person_passthrough = conditioning_mode in FIRST_PERSON_CONDITIONING_MODES
    texture_free_egocentric_arm_skeleton = (
        conditioning_mode in TEXTURE_FREE_EGOCENTRIC_ARM_SKELETON_MODES
    )
    oscar_gripper_scenario_proxy = conditioning_mode in OSCAR_GRIPPER_SCENARIO_PROXY_MODES
    projected_g1_skeleton_rgb_overlay = (
        conditioning_mode in PROJECTED_G1_SKELETON_RGB_OVERLAY_MODES
    )
    projected_g1_skeleton = conditioning_mode in ALL_PROJECTED_G1_SKELETON_MODES
    egocentric_arm_skeleton = (
        conditioning_mode in EGOCENTRIC_ARM_SKELETON_MODES
        or texture_free_egocentric_arm_skeleton
    )
    egocentric_arm_uses_background = (
        egocentric_arm_skeleton and not texture_free_egocentric_arm_skeleton
    )
    proxy_skeleton_overlay = conditioning_mode in PROXY_SKELETON_CONDITIONING_MODES
    background_required = (
        first_person_passthrough
        or egocentric_arm_uses_background
        or oscar_gripper_scenario_proxy
        or projected_g1_skeleton_rgb_overlay
    )
    background_conditioning = background_required or proxy_skeleton_overlay
    if background_required and not background_video:
        writer.release()
        raise ValueError("missing_review_video_for_first_person_conditioning")
    if background_video and background_conditioning:
        capture = cv2.VideoCapture(str(background_video))
        try:
            if capture.isOpened():
                source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                for index in range(len(sampled_rows)):
                    if source_count > 0:
                        source_index = round(
                            index * max(source_count - 1, 0) / max(len(sampled_rows) - 1, 1)
                        )
                        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        background_frames.append(None)
                        continue
                    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    resized, filled_pixels, total_pixels = _fill_near_black_void(
                        resized,
                        cv2=cv2,
                        np=np,
                        settings=background_settings,
                    )
                    background_void_fill_pixel_count += filled_pixels
                    background_total_pixel_count += total_pixels
                    if (
                        proxy_skeleton_overlay
                        or oscar_gripper_scenario_proxy
                        or projected_g1_skeleton_rgb_overlay
                    ):
                        resized = cv2.convertScaleAbs(
                            resized,
                            alpha=float(background_settings["background_alpha"]),
                            beta=0,
                        )
                    background_frames.append(resized)
                    background_frame_count += 1
        finally:
            capture.release()

    action_counts: dict[str, int] = {}
    fall_count = 0
    luma_means: list[float] = []
    luma_ranges: list[int] = []
    non_dark_fractions: list[float] = []
    projected_rows = list(projected_skeleton_rows or [])
    projected_sampled_rows = _sample_rows(projected_rows, len(sampled_rows)) if projected_rows else []
    projected_segments_drawn = 0
    projected_landmarks_drawn = 0
    for index, row in enumerate(sampled_rows):
        background = background_frames[index] if index < len(background_frames) else None
        if background is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = background.copy()
        action = _mapping(row.get("active_action"))
        action_type = _string(action.get("action_type")) or "unknown"
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
        if row.get("fall_detected") is True:
            fall_count += 1
        if first_person_passthrough:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luma_means.append(float(gray.mean()))
            luma_ranges.append(int(gray.max()) - int(gray.min()))
            non_dark_fractions.append(float(np.count_nonzero(gray > 12)) / float(width * height))
            writer.write(frame)
            continue
        if projected_g1_skeleton:
            if not projected_sampled_rows:
                writer.release()
                raise ValueError("missing_projected_g1_skeleton_trace_for_oscar_conditioning")
            skeleton_row = projected_sampled_rows[index]
            if not projected_g1_skeleton_rgb_overlay:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            landmarks_by_id: dict[str, tuple[int, int]] = {}
            for landmark in skeleton_row.get("landmarks", []) or []:
                if not isinstance(landmark, Mapping):
                    continue
                projection = _mapping(landmark.get("image_projection"))
                if not projection.get("available"):
                    continue
                u_px = _number(projection.get("u_px"), -1.0)
                v_px = _number(projection.get("v_px"), -1.0)
                if u_px < 0.0 or v_px < 0.0:
                    continue
                x = int(max(0, min(width - 1, round(u_px))))
                y = int(max(0, min(height - 1, round(v_px))))
                landmarks_by_id[_string(landmark.get("landmark_id"))] = (x, y)
            if not landmarks_by_id:
                writer.release()
                raise ValueError("projected_g1_skeleton_trace_has_no_projected_landmarks")
            segment_color = (70, 230, 255)
            joint_color = (255, 245, 150)
            hand_color = (110, 250, 190)
            for segment in skeleton_row.get("segments", []) or []:
                if not isinstance(segment, Mapping):
                    continue
                start = landmarks_by_id.get(_string(segment.get("from")))
                end = landmarks_by_id.get(_string(segment.get("to")))
                if start and end:
                    cv2.line(frame, start, end, segment_color, 5, cv2.LINE_AA)
                    projected_segments_drawn += 1
            for landmark_id, point in landmarks_by_id.items():
                color = hand_color if "hand" in landmark_id or "wrist" in landmark_id else joint_color
                radius = max(4, width // 96)
                cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
                cv2.circle(frame, point, radius + 2, (30, 35, 38), 1, cv2.LINE_AA)
                projected_landmarks_drawn += 1
            if action_type in {"waypoint", "base_velocity", "manipulation_contact"}:
                left = landmarks_by_id.get("left_hand") or landmarks_by_id.get("left_wrist")
                right = landmarks_by_id.get("right_hand") or landmarks_by_id.get("right_wrist")
                if left and right:
                    midpoint = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
                    arrow_end = (int(width * 0.52), int(height * 0.38))
                    cv2.arrowedLine(
                        frame,
                        midpoint,
                        arrow_end,
                        (72, 232, 255),
                        4,
                        cv2.LINE_AA,
                        tipLength=0.22,
                    )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luma_means.append(float(gray.mean()))
            luma_ranges.append(int(gray.max()) - int(gray.min()))
            non_dark_fractions.append(float(np.count_nonzero(gray > 12)) / float(width * height))
            writer.write(frame)
            continue
        if oscar_gripper_scenario_proxy:
            progress = index / max(len(sampled_rows) - 1, 1)
            reach = progress if action_type in {"base_velocity", "waypoint"} else 0.55
            if action_type == "manipulation_contact":
                reach = 1.0
            left_wrist = (
                int(width * (0.22 + 0.05 * reach)),
                int(height * (0.36 - 0.03 * reach)),
            )
            right_wrist = (
                int(width * (0.78 - 0.05 * reach)),
                int(height * (0.36 - 0.03 * reach)),
            )
            target_center = (
                int(width * (0.52 + 0.04 * reach)),
                int(height * (0.47 - 0.04 * reach)),
            )
            cue_color = (72, 232, 255)
            target_color = (0, 196, 255)
            grip_color = (248, 245, 230)
            mask_color = (96, 240, 180)
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (target_center[0] - int(width * 0.08), target_center[1] - int(height * 0.10)),
                (target_center[0] + int(width * 0.08), target_center[1] + int(height * 0.10)),
                (32, 180, 210),
                -1,
                cv2.LINE_AA,
            )
            frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)
            for wrist, side in ((left_wrist, -1), (right_wrist, 1)):
                cv2.ellipse(
                    frame,
                    wrist,
                    (max(22, width // 24), max(14, height // 36)),
                    0,
                    0,
                    360,
                    mask_color,
                    3,
                    cv2.LINE_AA,
                )
                palm = (wrist[0] + int(side * width * 0.035), wrist[1] - int(height * 0.015))
                cv2.line(frame, wrist, palm, grip_color, 4, cv2.LINE_AA)
                for finger_idx, angle in enumerate((-34, -15, 8, 28)):
                    finger_len = int(width * (0.035 - 0.004 * min(finger_idx, 2)))
                    dx = int(side * finger_len * math.cos(math.radians(angle)))
                    dy = int(finger_len * math.sin(math.radians(angle)) - height * 0.04)
                    tip = (palm[0] + dx, palm[1] + dy)
                    cv2.line(frame, palm, tip, grip_color, 3, cv2.LINE_AA)
                    cv2.circle(frame, tip, 3, cue_color, -1, cv2.LINE_AA)
            midpoint = ((left_wrist[0] + right_wrist[0]) // 2, (left_wrist[1] + right_wrist[1]) // 2)
            cv2.arrowedLine(
                frame,
                midpoint,
                target_center,
                cue_color,
                5,
                cv2.LINE_AA,
                tipLength=0.20,
            )
            cv2.circle(frame, target_center, max(8, width // 55), target_color, 4, cv2.LINE_AA)
            if action_type in {"stop", "inspect_look"}:
                cv2.line(
                    frame,
                    (target_center[0] - 18, target_center[1] - 18),
                    (target_center[0] + 18, target_center[1] + 18),
                    (48, 48, 255),
                    4,
                    cv2.LINE_AA,
                )
                cv2.line(
                    frame,
                    (target_center[0] + 18, target_center[1] - 18),
                    (target_center[0] - 18, target_center[1] + 18),
                    (48, 48, 255),
                    4,
                    cv2.LINE_AA,
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luma_means.append(float(gray.mean()))
            luma_ranges.append(int(gray.max()) - int(gray.min()))
            non_dark_fractions.append(float(np.count_nonzero(gray > 12)) / float(width * height))
            writer.write(frame)
            continue
        if egocentric_arm_skeleton:
            frame = (
                np.zeros((height, width, 3), dtype=np.uint8)
                if texture_free_egocentric_arm_skeleton
                else frame.copy()
            )
            arm_thickness = 7 if texture_free_egocentric_arm_skeleton else 3
            finger_thickness = 4 if texture_free_egocentric_arm_skeleton else 2
            joint_radius = 7 if texture_free_egocentric_arm_skeleton else 5
            fingertip_radius = 5 if texture_free_egocentric_arm_skeleton else 3
            target_thickness = 6 if texture_free_egocentric_arm_skeleton else 3
            arrow_thickness = 6 if texture_free_egocentric_arm_skeleton else 4
            progress = index / max(len(sampled_rows) - 1, 1)
            contact_reach = 1.0 if action_type == "manipulation_contact" else 0.35
            reach = progress * contact_reach
            left_shoulder = (int(width * -0.04), int(height * 0.82))
            left_elbow = (int(width * (0.16 + 0.04 * reach)), int(height * (0.58 - 0.03 * reach)))
            left_wrist = (int(width * (0.37 + 0.06 * reach)), int(height * (0.45 - 0.05 * reach)))
            right_shoulder = (int(width * 1.04), int(height * 0.82))
            right_elbow = (int(width * (0.84 - 0.04 * reach)), int(height * (0.58 - 0.03 * reach)))
            right_wrist = (int(width * (0.63 - 0.06 * reach)), int(height * (0.45 - 0.05 * reach)))
            arm_color = (80, 220, 255)
            joint_color = (255, 245, 120)
            target_color = (255, 185, 60)
            for start, end in [
                (left_shoulder, left_elbow),
                (left_elbow, left_wrist),
                (right_shoulder, right_elbow),
                (right_elbow, right_wrist),
            ]:
                cv2.line(frame, start, end, arm_color, arm_thickness, cv2.LINE_AA)
            finger_offsets = [(-22, -18), (-8, -28), (8, -28), (22, -18)]
            for wrist, side in ((left_wrist, -1), (right_wrist, 1)):
                cv2.circle(frame, wrist, joint_radius, joint_color, -1, cv2.LINE_AA)
                for dx, dy in finger_offsets:
                    finger_tip = (wrist[0] + dx * side, wrist[1] + dy)
                    cv2.line(frame, wrist, finger_tip, arm_color, finger_thickness, cv2.LINE_AA)
                    cv2.circle(frame, finger_tip, fingertip_radius, joint_color, -1, cv2.LINE_AA)
            target_center = (int(width * 0.52), int(height * 0.43))
            cv2.rectangle(
                frame,
                (target_center[0] - 38, target_center[1] - 28),
                (target_center[0] + 38, target_center[1] + 28),
                target_color,
                target_thickness,
                cv2.LINE_AA,
            )
            if action_type in {"waypoint", "base_velocity", "manipulation_contact"}:
                cv2.arrowedLine(
                    frame,
                    (int(width * 0.5), int(height * 0.36)),
                    (int(width * (0.5 + 0.08 * reach)), int(height * 0.28)),
                    (90, 220, 255),
                    arrow_thickness,
                    cv2.LINE_AA,
                    tipLength=0.25,
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luma_means.append(float(gray.mean()))
            luma_ranges.append(int(gray.max()) - int(gray.min()))
            non_dark_fractions.append(float(np.count_nonzero(gray > 12)) / float(width * height))
            writer.write(frame)
            continue
        root_x, root_y, root_z = _point_from_root(row)
        sx = int(width * 0.5 + (root_y - center_y) * scale)
        sy = int(height * 0.72 - (root_x - center_x) * scale)
        sy -= int((root_z - 0.78) * scale * 0.2)
        yaw = _number(row.get("root_yaw_rad"))
        color = (240, 240, 240) if row.get("fall_detected") is not True else (60, 60, 255)
        accent = (90, 220, 255)
        torso = max(42, int(height * 0.105))
        head_radius = max(8, int(height * 0.021))
        shoulder_half = max(18, int(width * 0.033))
        hip_half = max(13, int(width * 0.024))
        leg = max(35, int(height * 0.086))
        arm = max(32, int(height * 0.078))
        swing = math.sin(index * 0.45)
        yaw_dx = int(math.sin(yaw) * 18)
        hip = (sx, sy)
        neck = (sx + yaw_dx, sy - torso)
        head = (neck[0], neck[1] - torso // 2)
        left_shoulder = (neck[0] - shoulder_half, neck[1] + 6)
        right_shoulder = (neck[0] + shoulder_half, neck[1] + 6)
        left_hip = (sx - hip_half, sy)
        right_hip = (sx + hip_half, sy)
        arm_reach = 1.0 if action_type == "manipulation_contact" else 0.55
        if action_type in {"waypoint", "base_velocity"}:
            arm_phase = swing
        elif action_type == "inspect_look":
            arm_phase = math.sin(index * 0.16)
        else:
            arm_phase = 0.0
        left_hand = (
            int(left_shoulder[0] - arm * 0.42),
            int(left_shoulder[1] + arm * (0.45 - 0.18 * arm_phase)),
        )
        right_hand = (
            int(right_shoulder[0] + arm * arm_reach),
            int(right_shoulder[1] + arm * (0.2 + 0.15 * arm_phase)),
        )
        left_foot = (int(left_hip[0] - leg * 0.22), int(left_hip[1] + leg * (1.0 + 0.08 * swing)))
        right_foot = (int(right_hip[0] + leg * 0.22), int(right_hip[1] + leg * (1.0 - 0.08 * swing)))
        for start, end in [
            (hip, neck),
            (left_shoulder, right_shoulder),
            (left_hip, right_hip),
            (left_shoulder, left_hand),
            (right_shoulder, right_hand),
            (left_hip, left_foot),
            (right_hip, right_foot),
        ]:
            cv2.line(frame, start, end, color, 4, cv2.LINE_AA)
        cv2.circle(frame, head, head_radius, color, -1, cv2.LINE_AA)
        if action_type in {"waypoint", "base_velocity", "manipulation_contact"}:
            vx = _number(action.get("vx_mps"))
            vy = _number(action.get("vy_mps"))
            arrow_end = (
                int(sx + vy * 100),
                int(sy - torso - vx * 100),
            )
            cv2.arrowedLine(frame, (sx, sy - torso), arrow_end, accent, 3, cv2.LINE_AA, tipLength=0.25)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luma_means.append(float(gray.mean()))
        luma_ranges.append(int(gray.max()) - int(gray.min()))
        non_dark_fractions.append(float(np.count_nonzero(gray > 12)) / float(width * height))
        writer.write(frame)
    writer.release()
    low_signal_blockers: list[str] = []
    if (
        first_person_passthrough
        or egocentric_arm_uses_background
        or proxy_skeleton_overlay
        or projected_g1_skeleton_rgb_overlay
    ) and background_frame_count <= 0:
        low_signal_blockers.append("scene_overlay_conditioning_background_video_unreadable")
    if luma_ranges and max(luma_ranges) < 40:
        low_signal_blockers.append("proxy_conditioning_luma_range_too_low")
    if (
        non_dark_fractions
        and max(non_dark_fractions) < 0.03
        and not projected_g1_skeleton
    ):
        low_signal_blockers.append("proxy_conditioning_foreground_fraction_too_low")
    return {
        "path": str(output_path),
        "frame_count": len(sampled_rows),
        "fps": fps,
        "width": width,
        "height": height,
        "conditioning_mode": conditioning_mode,
        "conditioning_source": (
            "first_person_selected_review_video_with_simulated_robot_meshes"
            if first_person_passthrough
            else "projected_unitree_g1_mujoco_skeleton_over_selected_first_person_rgb"
            if projected_g1_skeleton_rgb_overlay
            else "official_style_2d_kinematic_skeleton_from_unitree_g1_mujoco_body_projection"
            if projected_g1_skeleton
            else "oscar_style_egocentric_rgb_gripper_action_proxy_from_selected_review_video_and_mujoco_trace"
            if oscar_gripper_scenario_proxy
            else "texture_free_egocentric_arm_hand_action_skeleton_from_mujoco_trace"
            if texture_free_egocentric_arm_skeleton
            else "selected_first_person_g1_mesh_video_with_egocentric_arm_hand_action_skeleton"
            if egocentric_arm_skeleton
            else "scene_overlay_proxy_skeleton_from_selected_review_video_and_mujoco_root_pose"
            if proxy_skeleton_overlay
            else "blueprint_proxy_skeleton_from_mujoco_root_pose_and_endpoint_actions"
        ),
        "first_person_review_video_passthrough": first_person_passthrough,
        "egocentric_arm_skeleton_rendered": egocentric_arm_skeleton,
        "projected_g1_skeleton_rendered": projected_g1_skeleton,
        "oscar_gripper_scenario_proxy_rendered": oscar_gripper_scenario_proxy,
        "texture_free_egocentric_arm_skeleton_rendered": (
            texture_free_egocentric_arm_skeleton
        ),
        "proxy_skeleton_overlay_drawn": bool(proxy_skeleton_overlay),
        "selected_review_video_background_used": bool(
            background_video
            and (
                first_person_passthrough
                or egocentric_arm_uses_background
                or proxy_skeleton_overlay
                or oscar_gripper_scenario_proxy
                or projected_g1_skeleton_rgb_overlay
            )
        ),
        "skeleton_stream_separate_from_rgb": bool(
            projected_g1_skeleton and not projected_g1_skeleton_rgb_overlay
        ),
        "skeleton_stream_texture_free": bool(
            projected_g1_skeleton and not projected_g1_skeleton_rgb_overlay
        ),
        "skeleton_stream_image_aligned_to_rgb": bool(projected_g1_skeleton),
        "first_rgb_frame_anchors_scene_and_robot_appearance": bool(projected_g1_skeleton),
        "alignment_contract": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": len(sampled_rows),
            "source_rgb_anchor": "first_frame",
            "skeleton_coordinates": "projected_pixel_landmarks",
        },
        "background_video_path": str(background_video) if background_video else None,
        "background_frame_count": background_frame_count,
        "background_preprocessing": {
            "near_black_void_fill_enabled": bool(
                background_settings["fill_near_black_void"]
            ),
            "near_black_threshold": int(background_settings["near_black_threshold"]),
            "void_fill_bgr": list(background_settings["void_fill_bgr"]),
            "background_alpha": float(background_settings["background_alpha"]),
            "background_alpha_applied_to_passthrough": False,
            "background_alpha_applied_to_gripper_proxy": bool(oscar_gripper_scenario_proxy),
            "background_alpha_applied_to_projected_g1_skeleton": bool(
                projected_g1_skeleton_rgb_overlay
            ),
            "void_fill_style": _string(background_settings.get("void_fill_style")),
            "void_fill_pixel_fraction": round(
                background_void_fill_pixel_count / background_total_pixel_count, 6
            )
            if background_total_pixel_count
            else 0.0,
            "preprocessing_applies_only_to_generated_conditioning_asset": True,
        },
        "simulated_g1_projected_kinematic_skeleton_available": bool(
            projected_g1_skeleton
        ),
        "true_robot_proprioceptive_skeleton_available": False,
        "projected_g1_skeleton_trace_row_count": len(projected_rows),
        "projected_g1_skeleton_projectable_row_count": _projected_skeleton_projectable_row_count(
            projected_rows
        ),
        "projected_g1_skeleton_landmark_draw_count": projected_landmarks_drawn,
        "projected_g1_skeleton_segment_draw_count": projected_segments_drawn,
        "action_type_counts": [
            {"action_type": key, "count": action_counts[key]} for key in sorted(action_counts)
        ],
        "fall_frame_count": fall_count,
        "visual_signal": {
            "status": "completed" if not low_signal_blockers else "warning_low_signal_proxy_conditioning",
            "mean_luma": round(sum(luma_means) / len(luma_means), 3) if luma_means else 0.0,
            "max_luma_range": max(luma_ranges) if luma_ranges else 0,
            "min_luma_range": min(luma_ranges) if luma_ranges else 0,
            "mean_non_dark_pixel_fraction": round(
                sum(non_dark_fractions) / len(non_dark_fractions), 6
            )
            if non_dark_fractions
            else 0.0,
            "blockers": low_signal_blockers,
        },
    }


def _render_rgb_context_video(
    *,
    review_video: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(review_video))
    if not capture.isOpened():
        raise ValueError("could_not_open_selected_review_video_for_rgb_context")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("cv2_video_writer_failed_for_oscar_rgb_context")
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    settings = _conditioning_background_settings()
    written = 0
    void_fill_pixel_count = 0
    total_pixel_count = 0
    luma_ranges: list[int] = []
    non_dark_fractions: list[float] = []
    try:
        for index in range(num_frames):
            if source_count > 0:
                source_index = round(index * max(source_count - 1, 0) / max(num_frames - 1, 1))
                capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            resized, filled_pixels, total_pixels = _fill_near_black_void(
                resized,
                cv2=cv2,
                np=np,
                settings=settings,
            )
            void_fill_pixel_count += filled_pixels
            total_pixel_count += total_pixels
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            luma_ranges.append(int(gray.max()) - int(gray.min()))
            non_dark_fractions.append(float(np.count_nonzero(gray > 12)) / float(width * height))
            writer.write(resized)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written <= 0:
        raise ValueError("could_not_decode_selected_review_video_for_rgb_context")
    return {
        "path": str(output_path),
        "source_review_video_path": str(review_video),
        "source_frame_count": source_count,
        "frame_count": written,
        "fps": fps,
        "width": width,
        "height": height,
        "normalized_for_oscar_inference": True,
        "near_black_void_fill_applied": bool(void_fill_pixel_count),
        "near_black_void_fill_pixel_fraction": round(
            void_fill_pixel_count / total_pixel_count, 6
        )
        if total_pixel_count
        else 0.0,
        "visual_signal": {
            "status": "completed",
            "max_luma_range": max(luma_ranges) if luma_ranges else 0,
            "min_luma_range": min(luma_ranges) if luma_ranges else 0,
            "mean_non_dark_pixel_fraction": round(
                sum(non_dark_fractions) / len(non_dark_fractions), 6
            )
            if non_dark_fractions
            else 0.0,
        },
    }


def _extract_first_frame(
    *,
    review_video: Path,
    output_path: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(review_video))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError("could_not_decode_selected_review_video_first_frame")
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    settings = _conditioning_background_settings()
    resized, filled_pixels, total_pixels = _fill_near_black_void(
        resized,
        cv2=cv2,
        np=np,
        settings=settings,
    )
    cv2.imwrite(str(output_path), resized)
    return {
        "path": str(output_path),
        "source_review_video_path": str(review_video),
        "width": width,
        "height": height,
        "near_black_void_fill_applied": bool(filled_pixels),
        "near_black_void_fill_pixel_fraction": round(
            filled_pixels / total_pixels, 6
        )
        if total_pixels
        else 0.0,
        "void_fill_style": _string(settings.get("void_fill_style")),
    }


def _materialize_oscar_input_package(
    *,
    rollout_manifest: Mapping[str, Any],
    work_dir: Path,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
) -> dict[str, Any]:
    package_dir = work_dir / "oscar_input"
    inputs = _mapping(rollout_manifest.get("inputs"))
    review_video = _selected_video_path(rollout_manifest)
    selected_video = _selected_video_row(rollout_manifest)
    trace_rows = _trace_rows(rollout_manifest)
    projected_skeleton_rows = _projected_skeleton_rows(rollout_manifest)
    conditioning_mode = _configured_conditioning_mode(projected_skeleton_rows)
    first_frame = _extract_first_frame(
        review_video=review_video,
        output_path=package_dir / "first_frame.png",
        width=width,
        height=height,
    )
    skeleton_video = _render_proxy_skeleton_video(
        trace_rows=trace_rows,
        output_path=package_dir / "blueprint_proxy_skeleton_conditioning.mp4",
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
        background_video=review_video,
        conditioning_mode=conditioning_mode,
        projected_skeleton_rows=projected_skeleton_rows,
    )
    skeleton_video_review_validation = validate_generated_mp4_for_review(
        Path(skeleton_video["path"])
    )
    skeleton_video_visual_smoke = visual_smoke_generated_rollouts_for_review(
        rollouts=[
            {
                "rollout_id": "oscar_conditioning_video_0001",
                "generated_video_path": skeleton_video["path"],
            }
        ],
        output_dir=work_dir / "oscar_input_conditioning_visual_review",
        generated_at=utc_now_iso(),
        require_review_quality_profile=False,
    )
    projected_g1_skeleton_rendered = bool(
        skeleton_video.get("projected_g1_skeleton_rendered")
    )
    conditioning_video_useful = _conditioning_video_model_input_useful(
        skeleton_video=skeleton_video,
        visual_smoke=skeleton_video_visual_smoke,
    )
    rgb_context_mode = _rgb_context_mode()
    rgb_video_used_for_latent_context = bool(
        not projected_g1_skeleton_rendered
        or skeleton_video.get("selected_review_video_background_used")
    )
    if rgb_context_mode == "always":
        rgb_video_used_for_latent_context = True
    elif rgb_context_mode == "never":
        rgb_video_used_for_latent_context = False
    rgb_context_video: dict[str, Any] | None = None
    if rgb_video_used_for_latent_context:
        rgb_context_video = _render_rgb_context_video(
            review_video=review_video,
            output_path=package_dir / "rgb_context.mp4",
            width=width,
            height=height,
            fps=fps,
            num_frames=num_frames,
        )
    manifest = {
        "schema_version": "blueprint_oscar_wam_input_package.v1",
        "status": "completed",
        "first_frame": first_frame,
        "skeleton_video": skeleton_video,
        "conditioning_video_review_validation": skeleton_video_review_validation,
        "conditioning_video_visual_smoke": skeleton_video_visual_smoke,
        "conditioning_video_decode_valid_for_review": (
            skeleton_video_review_validation.get("status") == "completed"
        ),
        "conditioning_video_visually_useful_for_model_input": conditioning_video_useful,
        "oscar_dual_stream_input_contract": {
            "first_rgb_frame_path": str(first_frame.get("path")),
            "skeleton_video_path": str(skeleton_video.get("path")),
            "separate_2d_skeleton_stream": bool(
                skeleton_video.get("skeleton_stream_separate_from_rgb")
            ),
            "skeleton_stream_texture_free": bool(
                skeleton_video.get("skeleton_stream_texture_free")
            ),
            "skeleton_stream_image_aligned_to_rgb": bool(
                skeleton_video.get("skeleton_stream_image_aligned_to_rgb")
            ),
            "first_rgb_frame_anchors_scene_and_robot_appearance": True,
            "full_rgb_video_required_for_oscar_inference": False,
            "width": width,
            "height": height,
            "fps": fps,
            "num_frames": num_frames,
        },
        "prompt": _task_prompt(rollout_manifest),
        "prompt_contract": {
            "task_specific_prompt_required": True,
            "generic_fallback_allowed": False,
            "future_ground_truth_used_to_construct_prompt": False,
            "oscar_public_source_revision": OSCAR_PUBLIC_SOURCE_REVISION,
        },
        "negative_prompt": OSCAR_DEFAULT_NEGATIVE_PROMPT,
        "negative_prompt_sha256": hashlib.sha256(
            OSCAR_DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
        ).hexdigest(),
        "num_frames": num_frames,
        "fps": fps,
        "height": height,
        "width": width,
        "source_review_video_path": str(review_video),
        "rgb_video": {
            "path": str(rgb_context_video["path"] if rgb_context_video else review_video),
            "source_review_video_path": str(review_video),
            "source": "selected_review_video_rgb_context",
            "used_for_oscar_rgb_latent_context": rgb_video_used_for_latent_context,
            "rgb_context_mode": rgb_context_mode,
            "omitted_by_rgb_context_mode": bool(
                rgb_context_mode == "never" and not rgb_video_used_for_latent_context
            ),
            "omitted_for_projected_g1_skeleton_conditioning": (
                projected_g1_skeleton_rendered
                and not rgb_video_used_for_latent_context
            ),
            "normalized_for_oscar_inference": bool(rgb_context_video),
            "frame_count": rgb_context_video.get("frame_count") if rgb_context_video else None,
            "fps": rgb_context_video.get("fps") if rgb_context_video else None,
            "height": rgb_context_video.get("height") if rgb_context_video else None,
            "width": rgb_context_video.get("width") if rgb_context_video else None,
            "visual_signal": rgb_context_video.get("visual_signal") if rgb_context_video else None,
        },
        "source_review_video": selected_video,
        "projected_skeleton_trace": {
            "path": _string(inputs.get("g1_projected_skeleton_trace_jsonl")) or None,
            "available": bool(projected_skeleton_rows),
            "used_for_conditioning": bool(
                skeleton_video.get("projected_g1_skeleton_rendered")
            ),
            "row_count": len(projected_skeleton_rows),
            "projectable_row_count": _projected_skeleton_projectable_row_count(
                projected_skeleton_rows
            ),
            "conditioning_source": "unitree_g1_mujoco_body_projection",
            "simulated_state_not_physical_robot_sensor_evidence": True,
        },
        "source_camera": selected_video.get("camera"),
        "scenario_eval_run_id": selected_video.get("scenario_eval_run_id"),
        "task_id": selected_video.get("task_id"),
        "spawn_id": selected_video.get("spawn_id"),
        "source_mujoco_endpoint_eval_job_dir": rollout_manifest.get(
            "source_mujoco_endpoint_eval_job_dir"
        ),
        "claim_boundary": {
            "skeleton_conditioning_is_proxy_from_mujoco_trace": bool(
                skeleton_video.get("proxy_skeleton_overlay_drawn")
                or skeleton_video.get("egocentric_arm_skeleton_rendered")
                or skeleton_video.get("oscar_gripper_scenario_proxy_rendered")
                or skeleton_video.get("projected_g1_skeleton_rendered")
            ),
            "projected_g1_skeleton_conditioning_used": bool(
                skeleton_video.get("projected_g1_skeleton_rendered")
            ),
            "projected_g1_skeleton_conditioning_is_simulated_mujoco_state": bool(
                skeleton_video.get("projected_g1_skeleton_rendered")
            ),
            "projected_g1_skeleton_conditioning_is_not_physical_robot_sensor_evidence": True,
            "scene_overlay_proxy_conditioning_is_not_true_robot_skeleton": bool(
                skeleton_video.get("proxy_skeleton_overlay_drawn")
            ),
            "egocentric_arm_skeleton_conditioning_is_texture_free_action_render": bool(
                skeleton_video.get("egocentric_arm_skeleton_rendered")
            ),
            "first_person_conditioning_uses_selected_review_video": bool(
                skeleton_video.get("first_person_review_video_passthrough")
                or (
                    skeleton_video.get("egocentric_arm_skeleton_rendered")
                    and skeleton_video.get("selected_review_video_background_used")
                )
                or (
                    skeleton_video.get("oscar_gripper_scenario_proxy_rendered")
                    and skeleton_video.get("selected_review_video_background_used")
                )
                or (
                    skeleton_video.get("projected_g1_skeleton_rendered")
                    and skeleton_video.get("selected_review_video_background_used")
                )
            ),
            "first_frame_uses_selected_review_video": True,
            "first_rgb_frame_anchors_scene_and_robot_appearance": True,
            "separate_2d_skeleton_stream_aligned_to_rgb": bool(
                skeleton_video.get("skeleton_stream_image_aligned_to_rgb")
            ),
            "skeleton_stream_is_texture_free": bool(
                skeleton_video.get("skeleton_stream_texture_free")
            ),
            "rgb_video_uses_selected_review_video": True,
            "rgb_video_used_for_oscar_rgb_latent_context": rgb_video_used_for_latent_context,
            "rgb_context_mode": rgb_context_mode,
            "rgb_video_arg_omitted_for_projected_g1_skeleton_conditioning": bool(
                projected_g1_skeleton_rendered
                and not rgb_video_used_for_latent_context
            ),
            "rgb_video_arg_omitted_by_rgb_context_mode": bool(
                rgb_context_mode == "never" and not rgb_video_used_for_latent_context
            ),
            "conditioning_video_uses_selected_first_person_g1_mesh_view": bool(
                (
                    skeleton_video.get("egocentric_arm_skeleton_rendered")
                    or skeleton_video.get("oscar_gripper_scenario_proxy_rendered")
                    or skeleton_video.get("projected_g1_skeleton_rendered")
                )
                and skeleton_video.get("selected_review_video_background_used")
            ),
            "simulated_g1_mesh_video_used_when_source_video_has_robot_meshes": bool(
                (
                    selected_video.get("hands_or_end_effectors_expected_in_view")
                    or selected_video.get(
                        "hands_or_end_effectors_expected_due_to_observation_pose"
                    )
                    or selected_video.get("egocentric_sensor_view")
                )
            ),
            "true_robot_proprioceptive_skeleton_available": False,
            "simulated_g1_projected_kinematic_skeleton_available": bool(
                skeleton_video.get("projected_g1_skeleton_rendered")
            ),
            "oscar_gripper_scenario_proxy_conditioning_is_support_asset_only": bool(
                skeleton_video.get("oscar_gripper_scenario_proxy_rendered")
            ),
            "conditioning_video_overlays_proxy_gripper_action_cues": bool(
                skeleton_video.get("oscar_gripper_scenario_proxy_rendered")
            ),
            "conditioning_video_preserves_selected_egocentric_rgb_context": bool(
                (
                    skeleton_video.get("oscar_gripper_scenario_proxy_rendered")
                    or skeleton_video.get("projected_g1_skeleton_rendered")
                )
                and skeleton_video.get("selected_review_video_background_used")
            ),
            "generated_input_is_not_model_output": True,
            "conditioning_visual_enhancement_applies_to_support_asset_only": True,
            "conditioning_video_visual_smoke_is_not_wam_output_success": True,
            "rgb_context_packaging_is_input_contract_not_rollout_quality_proof": True,
        },
    }
    manifest["claim_boundary"]["rgb_video_uses_selected_review_video"] = bool(
        rgb_video_used_for_latent_context
    )
    _write_json(work_dir / "oscar_wam_input_package_manifest.json", manifest)
    return manifest


def _cuda_library_path_candidates() -> list[str]:
    candidates = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    try:
        import torch  # type: ignore

        torch_package_dir = Path(torch.__file__).resolve().parent
        candidates.append(str(torch_package_dir / "lib"))
        site_root = torch_package_dir.parent
        candidates.extend(glob.glob(str(site_root / "nvidia" / "*" / "lib")))
        candidates.extend(glob.glob(str(site_root / "nvidia" / "*" / "lib64")))
    except Exception:
        pass
    candidates.extend(part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(Path(candidate).expanduser())
        if resolved in seen or not Path(resolved).exists():
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _runtime_env(source_root: Path) -> dict[str, str]:
    pythonpath = os.pathsep.join(
        part
        for part in [
            str(_repo_src_root()),
            str(source_root),
            os.environ.get("PYTHONPATH", ""),
        ]
        if part
    )
    env = {**os.environ, "PYTHONPATH": pythonpath}
    cuda_library_paths = _cuda_library_path_candidates()
    if cuda_library_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(cuda_library_paths)
    return env


def _run_import_probe(*, python: str, source_root: Path, timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    if platform.system() == "Darwin" and not shutil.which("nvidia-smi"):
        return {
            "schema_version": "oscar_wam_runtime_import_probe.v1",
            "status": "blocked",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 6),
            "module_available": {},
            "missing_modules": [],
            "torch_cuda_available": False,
            "platform_system": platform.system(),
            "blockers": [
                "blocked_oscar_linux_cuda_runtime_required",
                "blocked_oscar_requires_cuda_gpu_runtime",
            ],
            "stderr_size_bytes": 0,
            "stderr_omitted_to_avoid_secret_leakage": False,
        }
    probe = (
        "import importlib.util, json, platform\n"
        "mods=['torch','torchvision','cv2','decord','einops','diffusers','transformers','worldsim']\n"
        "available={m: bool(importlib.util.find_spec(m)) for m in mods}\n"
        "cuda=None\n"
        "try:\n"
        " import torch\n"
        " cuda=bool(torch.cuda.is_available())\n"
        "except Exception:\n"
        " cuda=False\n"
        "print(json.dumps({'module_available': available, 'torch_cuda_available': cuda, 'platform_system': platform.system()}))\n"
    )
    result = subprocess.run(
        [python, "-c", probe],
        cwd=str(source_root),
        env=_runtime_env(source_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    payload: dict[str, Any] = {}
    if result.stdout.strip():
        try:
            value = json.loads(result.stdout)
            payload = dict(value) if isinstance(value, Mapping) else {}
        except json.JSONDecodeError:
            payload = {}
    module_available = _mapping(payload.get("module_available"))
    missing = [key for key, available in module_available.items() if available is False]
    blockers: list[str] = []
    if result.returncode != 0 or not module_available:
        blockers.append("blocked_oscar_runtime_import_probe_failed")
    if missing:
        blockers.append("blocked_missing_oscar_runtime_imports")
    if payload.get("torch_cuda_available") is not True:
        blockers.append("blocked_oscar_requires_cuda_gpu_runtime")
    return {
        "schema_version": "oscar_wam_runtime_import_probe.v1",
        "status": "completed" if not blockers else "blocked",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "module_available": module_available,
        "missing_modules": missing,
        "torch_cuda_available": payload.get("torch_cuda_available"),
        "platform_system": payload.get("platform_system"),
        "blockers": blockers,
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
    }


def _redacted_argv(argv: Sequence[str], checkpoint: Path) -> list[str]:
    checkpoint_text = str(checkpoint)
    return ["<checkpoint_path_configured>" if item == checkpoint_text else item for item in argv]


def _run_oscar(
    *,
    python: str,
    source_root: Path,
    checkpoint: Path,
    package_manifest: Mapping[str, Any],
    output_video: Path,
    timeout_seconds: float,
    num_steps: int,
    guidance: float,
    seed: int,
) -> dict[str, Any]:
    started = time.monotonic()
    argv = [
        python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "inference/inference_oscar.py",
        "--checkpoint",
        str(checkpoint),
        "--first-frame",
        _string(_mapping(package_manifest.get("first_frame")).get("path")),
        "--skeleton-video",
        _string(_mapping(package_manifest.get("skeleton_video")).get("path")),
        "--start-frame",
        "0",
        "--prompt",
        _string(package_manifest.get("prompt")),
        "--negative-prompt",
        _string(package_manifest.get("negative_prompt") or OSCAR_DEFAULT_NEGATIVE_PROMPT),
        "--num-steps",
        str(num_steps),
        "--guidance",
        str(guidance),
        "--seed",
        str(seed),
        "--num-frames",
        str(int(package_manifest.get("num_frames") or DEFAULT_NUM_FRAMES)),
        "--height",
        str(int(package_manifest.get("height") or DEFAULT_HEIGHT)),
        "--width",
        str(int(package_manifest.get("width") or DEFAULT_WIDTH)),
        "--fps",
        str(float(package_manifest.get("fps") or DEFAULT_FPS)),
        "--output",
        str(output_video),
    ]
    rgb_package = _mapping(package_manifest.get("rgb_video"))
    rgb_video = Path(_string(rgb_package.get("path"))).expanduser()
    projected_conditioning_used = _package_uses_projected_g1_skeleton(package_manifest)
    rgb_context_allowed = bool(
        rgb_package.get("used_for_oscar_rgb_latent_context") is not False
        if not projected_conditioning_used
        else rgb_package.get("used_for_oscar_rgb_latent_context") is True
    )
    if rgb_context_allowed and rgb_video.is_file():
        argv.extend(["--rgb-video", str(rgb_video)])
    stale_output_removed = False
    if output_video.exists():
        try:
            output_video.unlink()
            stale_output_removed = True
        except OSError as exc:
            return {
                "schema_version": "oscar_wam_subprocess_result.v1",
                "status": "blocked",
                "returncode": None,
                "duration_seconds": round(time.monotonic() - started, 6),
                "argv_redacted": _redacted_argv(argv, checkpoint),
                "stdout_size_bytes": 0,
                "stderr_size_bytes": 0,
                "stderr_omitted_to_avoid_secret_leakage": False,
                "stale_output_removed_before_launch": False,
                "blockers": [f"oscar_inference_stale_output_unlink_failed:{type(exc).__name__}"],
            }
    try:
        result = subprocess.run(
            argv,
            cwd=str(source_root),
            env=_runtime_env(source_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.output or ""
        stderr = exc.stderr or ""
        return {
            "schema_version": "oscar_wam_subprocess_result.v1",
            "status": "blocked",
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "duration_seconds": round(time.monotonic() - started, 6),
            "argv_redacted": _redacted_argv(argv, checkpoint),
            "stdout_size_bytes": len(stdout),
            "stderr_size_bytes": len(stderr),
            "stderr_omitted_to_avoid_secret_leakage": bool(stderr),
            "stale_output_removed_before_launch": stale_output_removed,
            "blockers": ["oscar_inference_command_timeout"],
        }
    return {
        "schema_version": "oscar_wam_subprocess_result.v1",
        "status": "completed" if result.returncode == 0 else "blocked",
        "returncode": result.returncode,
        "timed_out": False,
        "timeout_seconds": timeout_seconds,
        "duration_seconds": round(time.monotonic() - started, 6),
        "argv_redacted": _redacted_argv(argv, checkpoint),
        "stdout_size_bytes": len(result.stdout or ""),
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
        "stale_output_removed_before_launch": stale_output_removed,
        "configured_inference_steps": num_steps,
        "blockers": [] if result.returncode == 0 else ["oscar_inference_command_nonzero"],
    }


def _rollout_payload(
    *,
    package_manifest: Mapping[str, Any],
    checkpoint: Path,
    source_root: Path,
    subprocess_detail: Mapping[str, Any],
    output_video: Path,
    official_release: Mapping[str, Any] | None = None,
    source_url: str = "",
    source_ref: str = "",
    checkpoint_repo: str = OFFICIAL_OSCAR_HF_REPO,
    checkpoint_revision: str = "",
) -> dict[str, Any]:
    video_validation = validate_generated_mp4_for_review(output_video)
    video_reviewable = video_validation.get("status") == "completed"
    subprocess_completed = subprocess_detail.get("status") == "completed"
    official_release_payload = dict(
        official_release
        or official_release_contract(
            source_url=source_url or (_source_root_origin_url(source_root) or ""),
            source_ref=source_ref or (_source_root_commit(source_root) or ""),
            hf_repo=checkpoint_repo,
            hf_revision=checkpoint_revision or (_checkpoint_revision_from_path(checkpoint) or ""),
        )
    )
    official_release_match = official_release_payload.get("official_release_match") is True
    rollouts = (
        [
            {
                "rollout_id": "oscar_wam_rollout_0001",
                "policy_id": ADAPTER_ID,
                "model_candidate": "oscar_wam",
                "generated_video_path": str(output_video),
                "generated_video_sha256": sha256_file(output_video),
                "source_review_video_path": package_manifest.get("source_review_video_path"),
                "source_camera": package_manifest.get("source_camera"),
                "scenario_eval_run_id": package_manifest.get("scenario_eval_run_id"),
                "task_id": package_manifest.get("task_id"),
                "spawn_id": package_manifest.get("spawn_id"),
                "model_rollout_confidence": None,
                "generated_rollout_termination_reason": "oscar_inference_command_completed",
                "success_label_source": "generated_video_requires_review",
                "generated_video_review_validation": video_validation,
            }
        ]
        if video_reviewable and subprocess_completed
        else []
    )
    validation_blockers = [str(item) for item in video_validation.get("blockers", []) if str(item)]
    if rollouts:
        blockers = []
    else:
        blockers = []
        if not subprocess_completed:
            blockers.append("blocked_oscar_inference_command_not_completed")
        blockers.append(
            "blocked_generated_oscar_mp4_not_reviewable"
            if output_video.is_file()
            else "blocked_no_generated_oscar_mp4"
        )
        blockers.extend(validation_blockers)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if rollouts else "blocked",
        "adapter_id": ADAPTER_ID,
        "rollouts": rollouts,
        "generated_video_count": len(rollouts),
        "generated_video_review_validation": video_validation,
        "model_provenance": {
            "candidate": "oscar_wam",
            "source_root": str(source_root),
            "source_url": source_url or (_source_root_origin_url(source_root) or None),
            "source_ref": source_ref or (_source_root_commit(source_root) or None),
            "checkpoint_repo": checkpoint_repo,
            "checkpoint_revision": checkpoint_revision
            or (_checkpoint_revision_from_path(checkpoint) or None),
            "checkpoint_path": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "oscar_public_inference_entrypoint": str(source_root / "inference" / "inference_oscar.py"),
            "official_oscar_release": official_release_payload,
        },
        "official_oscar_release": official_release_payload,
        "input_package": dict(package_manifest),
        "oscar_subprocess": dict(subprocess_detail),
        "blockers": blockers,
        "fresh_model_command_executed_this_invocation": bool(rollouts and subprocess_completed),
        "fresh_model_run_steps": len(rollouts) if subprocess_completed else 0,
        "configured_inference_steps_per_model_run": int(subprocess_detail.get("configured_inference_steps") or 0) if rollouts and subprocess_completed else 0,
        "fresh_model_run_claimed": bool(
            rollouts
            and subprocess_detail.get("status") == "completed"
            and official_release_match
        ),
        "learned_wam_model_ran": bool(
            rollouts
            and subprocess_detail.get("status") == "completed"
            and official_release_match
        ),
        "truth_boundary": {
            "generated_video_is_model_output": bool(
                rollouts and subprocess_detail.get("status") == "completed"
            ),
            "official_oscar_source_and_checkpoint_pinned": official_release_match,
            "official_oscar_release_match_required_for_learned_wam_claim": True,
            "generated_outputs_support_artifacts_not_deployment_proof": True,
            "generated_rollout_not_physical_robot_proof": True,
            "generated_success_label_requires_external_vlm_or_human_judge": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--source-url",
        default=os.getenv("BLUEPRINT_OSCAR_WAM_SOURCE_URL", ""),
        help="Origin URL for the OSCAR source tree when git metadata is unavailable.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-repo",
        default=os.getenv("BLUEPRINT_OSCAR_WAM_HF_REPO", OFFICIAL_OSCAR_HF_REPO),
    )
    parser.add_argument(
        "--checkpoint-revision",
        default=(
            os.getenv("BLUEPRINT_OSCAR_WAM_HF_REVISION")
            or os.getenv("BLUEPRINT_WAM_MODEL_CHECKPOINT_REVISION")
            or ""
        ),
    )
    parser.add_argument(
        "--allow-experimental-oscar-version",
        action="store_true",
        default=_env_flag(ALLOW_EXPERIMENTAL_OSCAR_VERSION_ENV, default=False),
        help=(
            "Permit a non-official OSCAR source/checkpoint for diagnostics. "
            "Outputs remain experimental and do not set the learned OSCAR claim booleans."
        ),
    )
    parser.add_argument("--python", default=os.getenv("BLUEPRINT_OSCAR_WAM_PYTHON") or sys.executable)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--num-frames", type=int, default=int(os.getenv("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", str(DEFAULT_NUM_FRAMES))))
    parser.add_argument("--height", type=int, default=int(os.getenv("BLUEPRINT_OSCAR_WAM_HEIGHT", str(DEFAULT_HEIGHT))))
    parser.add_argument("--width", type=int, default=int(os.getenv("BLUEPRINT_OSCAR_WAM_WIDTH", str(DEFAULT_WIDTH))))
    parser.add_argument("--fps", type=float, default=float(os.getenv("BLUEPRINT_OSCAR_WAM_FPS", str(DEFAULT_FPS))))
    parser.add_argument("--num-steps", type=int, default=int(os.getenv("BLUEPRINT_OSCAR_WAM_NUM_STEPS", "35")))
    parser.add_argument("--guidance", type=float, default=float(os.getenv("BLUEPRINT_OSCAR_WAM_GUIDANCE", "6.0")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("BLUEPRINT_OSCAR_WAM_SEED", "42")))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("BLUEPRINT_OSCAR_WAM_TIMEOUT_SECONDS", "3600")))
    parser.add_argument("--probe-only", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    source_root = (
        args.source_root.expanduser().resolve() if args.source_root else _source_root_from_env()
    )
    checkpoint = (
        args.checkpoint.expanduser().resolve() if args.checkpoint else _checkpoint_from_env()
    )
    output_path = Path(os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")).resolve()
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output_path.parent / "oscar_wam_command_workspace"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    blockers: list[str] = []
    if source_root is None:
        blockers.append("blocked_missing_oscar_source_root")
    elif not (source_root / "inference" / "inference_oscar.py").is_file():
        blockers.append("blocked_missing_oscar_inference_entrypoint")
    if checkpoint is None:
        blockers.append("blocked_missing_oscar_checkpoint")
    elif not checkpoint.exists():
        blockers.append("blocked_configured_oscar_checkpoint_path_missing")
    if not shutil.which(args.python) and not Path(args.python).expanduser().is_file():
        blockers.append("blocked_configured_python_missing")

    source_ref = _source_root_commit(source_root) or ""
    source_url = _string(args.source_url) or (_source_root_origin_url(source_root) or "")
    checkpoint_revision = (
        _string(args.checkpoint_revision)
        or (_checkpoint_revision_from_path(checkpoint) if checkpoint else "")
        or ""
    )
    checkpoint_repo = _string(args.checkpoint_repo) or OFFICIAL_OSCAR_HF_REPO
    official_release = official_release_contract(
        source_url=source_url,
        source_ref=source_ref,
        hf_repo=checkpoint_repo,
        hf_revision=checkpoint_revision,
    )
    experimental_oscar_version_allowed = bool(args.allow_experimental_oscar_version)
    official_version_blockers = (
        []
        if experimental_oscar_version_allowed
        or args.probe_only
        or source_root is None
        or checkpoint is None
        or not checkpoint.exists()
        else official_release_blockers(official_release)
    )
    blockers.extend(official_version_blockers)

    package_manifest: dict[str, Any] = {}
    if not blockers:
        try:
            rollout_input = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_INPUT"]).expanduser().resolve()
            rollout_manifest = _read_json(rollout_input)
            package_manifest = _materialize_oscar_input_package(
                rollout_manifest=rollout_manifest,
                work_dir=work_dir,
                width=args.width,
                height=args.height,
                fps=args.fps,
                num_frames=args.num_frames,
            )
        except Exception as exc:
            blockers.append(f"blocked_oscar_input_package_materialization_failed:{type(exc).__name__}")

    if blockers:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "adapter_id": ADAPTER_ID,
            "blockers": blockers,
            "source_root": str(source_root) if source_root else None,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "model_provenance": {
                "candidate": "oscar_wam",
                "source_root": str(source_root) if source_root else None,
                "source_url": source_url or None,
                "source_ref": source_ref or None,
                "checkpoint_path": str(checkpoint) if checkpoint else None,
                "checkpoint_repo": checkpoint_repo,
                "checkpoint_revision": checkpoint_revision or None,
                "official_oscar_release": official_release,
            },
            "official_oscar_release": official_release,
            "experimental_oscar_version_allowed": experimental_oscar_version_allowed,
            "input_package": package_manifest or None,
            "truth_boundary": {
                "official_oscar_source_and_checkpoint_pinned": bool(
                    official_release.get("official_release_match") is True
                ),
                "official_oscar_release_match_required_for_learned_wam_claim": True,
                "blocked_output_is_not_model_proof": True,
                "generated_outputs_support_artifacts_not_deployment_proof": True,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        _write_json(output_path, payload)
        return payload

    assert source_root is not None
    assert checkpoint is not None
    probe = _run_import_probe(
        python=args.python,
        source_root=source_root,
        timeout_seconds=min(args.timeout_seconds, 120.0),
    )
    _write_json(work_dir / "oscar_wam_import_probe.json", probe)
    if args.probe_only or probe["status"] != "completed":
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": probe["status"],
            "adapter_id": ADAPTER_ID,
            "probe_only": bool(args.probe_only),
            "source_root": str(source_root),
            "checkpoint_path": str(checkpoint),
            "model_provenance": {
                "candidate": "oscar_wam",
                "source_root": str(source_root),
                "source_url": source_url or None,
                "source_ref": source_ref or None,
                "checkpoint_path": str(checkpoint),
                "checkpoint_repo": checkpoint_repo,
                "checkpoint_revision": checkpoint_revision or None,
                "official_oscar_release": official_release,
            },
            "official_oscar_release": official_release,
            "experimental_oscar_version_allowed": experimental_oscar_version_allowed,
            "input_package": package_manifest,
            "import_probe": probe,
            "blockers": probe.get("blockers", []),
            "truth_boundary": {
                "official_oscar_source_and_checkpoint_pinned": bool(
                    official_release.get("official_release_match") is True
                ),
                "official_oscar_release_match_required_for_learned_wam_claim": True,
                "probe_only_is_not_model_execution_proof": bool(args.probe_only),
                "generated_outputs_support_artifacts_not_deployment_proof": True,
            },
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        _write_json(output_path, payload)
        return payload

    output_video = work_dir / "oscar_generated_rollout.mp4"
    subprocess_detail = _run_oscar(
        python=args.python,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest=package_manifest,
        output_video=output_video,
        timeout_seconds=args.timeout_seconds,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
    )
    payload = _rollout_payload(
        package_manifest=package_manifest,
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail=subprocess_detail,
        output_video=output_video,
        official_release=official_release,
        source_url=source_url,
        source_ref=source_ref,
        checkpoint_repo=checkpoint_repo,
        checkpoint_revision=checkpoint_revision,
    )
    payload["experimental_oscar_version_allowed"] = experimental_oscar_version_allowed
    if subprocess_detail["status"] != "completed" and not payload["rollouts"]:
        payload["status"] = "blocked"
        payload["blockers"] = list(subprocess_detail.get("blockers") or payload["blockers"])
    _write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(argv)
    except Exception as exc:
        output_path = Path(os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")).resolve()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "adapter_id": ADAPTER_ID,
            "blockers": [f"oscar_wam_adapter_exception:{type(exc).__name__}"],
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        _write_json(output_path, payload)
    print(json.dumps({"adapter_id": ADAPTER_ID, "status": payload.get("status")}, sort_keys=True))
    return 0 if payload.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
