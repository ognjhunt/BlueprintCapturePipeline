"""Canonical dynamic episode/render settings shared by G1 kitchen adapters."""

from __future__ import annotations


def build_episode_render_settings(
    *,
    steps: int,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    per_scenario_seconds: int,
    dynamic_episode_termination: bool,
    episode_max_steps: int,
) -> dict[str, int | bool | None]:
    """Use a frame expectation only for explicitly fixed-horizon diagnostics."""

    return {
        "steps": int(steps),
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "warmup_frames": int(warmup_frames),
        "per_scenario_seconds": int(per_scenario_seconds),
        "dynamic_episode_termination": bool(dynamic_episode_termination),
        "episode_max_steps": int(episode_max_steps or 0),
        "expected_frame_count_per_scenario": (None if dynamic_episode_termination else int(steps)),
    }
