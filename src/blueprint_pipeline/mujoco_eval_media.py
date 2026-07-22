"""Media sampling, timing, and inspection contracts for MuJoCo evaluation lanes."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_VIDEO_FRAME_STRIDE_STEPS = 8
DEFAULT_REVIEW_VIDEO_FPS = 60


def episode_frame_steps(
    *,
    steps_per_episode: int,
    render_frame_count: int,
    video_frame_stride_steps: int,
) -> tuple[list[int], str, int]:
    steps = max(1, int(steps_per_episode))
    if int(render_frame_count) > 0:
        if int(render_frame_count) <= 1:
            return [0], "fixed_sample_count", steps
        stride = max(1, steps // max(1, int(render_frame_count) - 1))
        frame_steps = sorted(
            {min(steps - 1, step * stride) for step in range(int(render_frame_count))}
        )
        return frame_steps, "fixed_sample_count", stride
    stride = max(1, int(video_frame_stride_steps))
    frame_steps = list(range(0, steps, stride))
    if frame_steps[-1] != steps - 1:
        frame_steps.append(steps - 1)
    return frame_steps, "full_episode_stride", stride


def video_output_fps(*, requested_fps: int, timestep: float, stride_steps: int) -> int:
    if int(requested_fps) > 0:
        return int(requested_fps)
    sim_seconds_per_frame = max(float(timestep) * max(1, int(stride_steps)), 1e-6)
    return max(1, int(round(1.0 / sim_seconds_per_frame)))


def video_timing_contract(
    *,
    requested_fps: int,
    encoded_fps: int,
    timestep: float,
    stride_steps: int,
    physics_frame_count: int,
    encoded_frame_count: int,
) -> dict[str, Any]:
    sim_seconds_per_frame = max(float(timestep) * max(1, int(stride_steps)), 1e-9)
    expected_sim_time_fps = max(1, int(round(1.0 / sim_seconds_per_frame)))
    physics_duration_s = max(0.0, float(physics_frame_count) * sim_seconds_per_frame)
    encoded_duration_s = (
        max(0.0, float(encoded_frame_count) / max(1, int(encoded_fps)))
        if encoded_frame_count
        else 0.0
    )
    playback_scale = encoded_duration_s / physics_duration_s if physics_duration_s > 0 else None
    fixed_fps_forced = int(requested_fps) > 0
    slow_motion = bool(fixed_fps_forced and playback_scale is not None and playback_scale > 1.2)
    return {
        "requested_fps": int(requested_fps),
        "encoded_video_fps": int(encoded_fps),
        "expected_sim_time_fps_for_stride": expected_sim_time_fps,
        "sim_seconds_per_rendered_frame": round(sim_seconds_per_frame, 9),
        "physics_duration_s": round(physics_duration_s, 9),
        "encoded_duration_estimate_s": round(encoded_duration_s, 9),
        "playback_time_scale_vs_sim": round(playback_scale, 6)
        if playback_scale is not None
        else None,
        "fps_zero_used_for_sim_time_playback": not fixed_fps_forced,
        "fixed_fps_forced_by_user": fixed_fps_forced,
        "video_playback_may_look_slow_motion": slow_motion,
        "slow_motion_reason": (
            "fixed_fps_lower_than_mujoco_step_rate_for_captured_frames" if slow_motion else None
        ),
    }


def review_video_sampling_contract(
    *,
    fps: int,
    timestep: float,
    video_frame_stride_steps: int,
    render_frame_count: int,
    extend_terminal_frame_for_review: bool,
) -> dict[str, Any]:
    stride = max(1, int(video_frame_stride_steps))
    expected_sim_time_fps = video_output_fps(
        requested_fps=0,
        timestep=timestep,
        stride_steps=stride,
    )
    captures_every_step = stride == 1
    fixed_fps = int(fps) > 0
    default_stride = int(render_frame_count) <= 0 and stride == DEFAULT_VIDEO_FRAME_STRIDE_STEPS
    nominal_realtime_review_mp4 = bool(default_stride and int(fps) == DEFAULT_REVIEW_VIDEO_FPS)
    if nominal_realtime_review_mp4:
        sampling_mode = "nominal_realtime_stride_review"
    elif captures_every_step and not fixed_fps:
        sampling_mode = "every_sim_step_sim_time_review"
    elif captures_every_step and fixed_fps:
        sampling_mode = "every_sim_step_fixed_fps_debug_slow_motion"
    elif int(render_frame_count) > 0:
        sampling_mode = "fixed_sample_count_review"
    else:
        sampling_mode = "custom_stride_review"
    return {
        "schema_version": "review_video_sampling_contract.v1",
        "sampling_mode": sampling_mode,
        "sample_every_n_sim_steps": stride,
        "captures_every_mujoco_step": captures_every_step,
        "captures_bounded_stride_frames": not captures_every_step,
        "mujoco_timestep_s": round(float(timestep), 9),
        "sim_seconds_per_rendered_frame": round(float(timestep) * stride, 9),
        "expected_sim_time_fps_for_stride": expected_sim_time_fps,
        "requested_or_default_fps": int(fps),
        "default_review_video_fps": DEFAULT_REVIEW_VIDEO_FPS,
        "nominal_realtime_review_mp4": nominal_realtime_review_mp4,
        "recommended_for_matrix_runs": nominal_realtime_review_mp4,
        "every_frame_at_fixed_60fps_is_debug_slow_motion": bool(captures_every_step and fixed_fps),
        "why_not_every_frame_by_default": (
            "MuJoCo steps at 0.002s by default; encoding every step at fixed 60fps "
            "turns simulator time into slow-motion review video. The default samples "
            "every 8 sim steps and encodes at 60fps, which is close to real-time."
        ),
        "terminal_failure_frame_hold_enabled": bool(extend_terminal_frame_for_review),
        "review_video_stops_at_terminal_failure_by_default": not bool(
            extend_terminal_frame_for_review
        ),
    }


def ffprobe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"path": str(path), "status": "not_checked", "reason": "ffprobe_unavailable"}
    if not path.is_file():
        return {"path": str(path), "status": "not_checked", "reason": "missing_video"}
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,duration,width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {"path": str(path), "status": "blocked", "stderr": (result.stderr or "")[-500:]}
    payload = json.loads(result.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    frame_count = stream.get("nb_frames")
    frame_count_int = int(frame_count) if str(frame_count).isdigit() else None
    duration = float(stream.get("duration") or 0.0)
    return {
        "path": str(path),
        "status": "complete" if duration > 0 and (frame_count_int or 0) > 0 else "blocked",
        "duration_s": duration,
        "frame_count": frame_count_int,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
    }


def counts_by_key(attempts: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        value = str(attempt.get(key) or "unknown")
        row = grouped.setdefault(
            value,
            {"id": value, "attempted": 0, "passed": 0, "failed": 0, "blocked": 0},
        )
        row["attempted"] += 1
        if attempt.get("status") == "blocked":
            row["blocked"] += 1
        elif attempt.get("success") is True:
            row["passed"] += 1
        else:
            row["failed"] += 1
    return [grouped[group_id] for group_id in sorted(grouped)]
