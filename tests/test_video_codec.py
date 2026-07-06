"""The video-codec gate must FAIL on CI, not silently skip.

A silently-skipped video test on CI masks real delivery-video failures (the
"PTDP flake"). Locally without the codec, skipping is fine; when a codec is
required (CI / BLUEPRINT_REQUIRE_VIDEO_CODEC) an unavailable codec is a hard
failure so the delivery-video path is actually exercised.
"""

from __future__ import annotations

import pytest

from tests.video_codec import require_video_codec_or_skip


def test_fails_when_codec_required_via_explicit_flag():
    with pytest.raises(pytest.fail.Exception):
        require_video_codec_or_skip(
            "cv2 mp4 writer unavailable", env={"BLUEPRINT_REQUIRE_VIDEO_CODEC": "1"}
        )


def test_fails_when_codec_required_via_ci():
    with pytest.raises(pytest.fail.Exception):
        require_video_codec_or_skip("cv2 mp4 writer unavailable", env={"CI": "true"})


def test_skips_when_codec_not_required():
    with pytest.raises(pytest.skip.Exception):
        require_video_codec_or_skip("cv2 mp4 writer unavailable", env={})
