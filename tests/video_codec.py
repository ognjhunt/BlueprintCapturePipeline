"""Shared gate for video tests: a missing codec FAILS on CI, skips locally.

A silently-skipped video test on CI masks real delivery-video failures. Call
``require_video_codec_or_skip(reason)`` where a video test would otherwise
``pytest.skip`` on a missing cv2 mp4 writer: it hard-fails when a codec is
required (CI or BLUEPRINT_REQUIRE_VIDEO_CODEC set) and skips otherwise.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

import pytest

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in _TRUTHY if value is not None else False


def video_codec_required(env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return _truthy(env.get("BLUEPRINT_REQUIRE_VIDEO_CODEC")) or _truthy(env.get("CI"))


def require_video_codec_or_skip(
    reason: str, *, env: Optional[Mapping[str, str]] = None
) -> None:
    if video_codec_required(env):
        pytest.fail(
            f"{reason} — but a video codec is required "
            "(CI / BLUEPRINT_REQUIRE_VIDEO_CODEC set); the delivery-video path "
            "must be exercised, not skipped"
        )
    pytest.skip(reason)
