"""The movable-furniture analysis admits the video the measurement bound, in the measurement's own digest form.

capture-447ae4f4 2026-11-04: ``robot_access_measurement`` binds the source video as ``sha256:<hex>``
(``_sha``) while ``analyze_movable_obstacles`` compared it against a bare hex digest, so every
measurement was held as ``website_gemini_retained_source_required`` before any spend. The earlier tests
replaced the analysis with a fake, so the real comparison never ran.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import clean_plate_removal_analysis_gemini as gemini
from blueprint_pipeline import robot_access_measurement as measurement
from blueprint_pipeline import website_gemini_receipts as receipts
from blueprint_pipeline.core.common import sha256_file


def _analyze(tmp_path: Path, monkeypatch, video: Path, source_digest: str) -> list[dict]:
    calls: list[dict] = []

    def retained(**kwargs):
        calls.append(kwargs)
        return {"schema_version": "movable_obstacle_analysis.v1", "targets": [], "blockers": []}

    # The paid call itself is never made here: only what reaches it.
    monkeypatch.setattr(receipts, "retained_gemini_call", retained)
    gemini.analyze_movable_obstacles(video_path=video, task_context={"capture_id": "capture-1"},
                                     output_root=tmp_path / "analysis",
                                     binding={"source_digest": source_digest, "model": "m", "processing": "static"},
                                     maximum_cost_usd=0.04)
    return calls


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "raw" / "walkthrough.mp4"
    path.parent.mkdir()
    path.write_bytes(b"walkthrough bytes")
    return path


def test_the_measurements_prefixed_digest_reaches_the_retained_call(tmp_path, monkeypatch, video):
    digest = measurement._sha(video)
    assert digest.startswith("sha256:")
    [call] = _analyze(tmp_path, monkeypatch, video, digest)
    assert call["binding"]["source_digest"] == digest and call["maximum_cost_usd"] == 0.04


def test_a_bare_hex_digest_still_reaches_the_retained_call(tmp_path, monkeypatch, video):
    assert len(_analyze(tmp_path, monkeypatch, video, sha256_file(video))) == 1


@pytest.mark.parametrize("digest", ["sha256:" + "0" * 64, "", "sha256:"])
def test_another_or_missing_digest_is_refused_before_any_call(tmp_path, monkeypatch, video, digest):
    with pytest.raises(ValueError, match="website_gemini_retained_source_required"):
        _analyze(tmp_path, monkeypatch, video, digest)


def test_a_missing_video_is_refused_before_any_call(tmp_path, monkeypatch, video):
    digest = measurement._sha(video)
    video.unlink()
    with pytest.raises(ValueError, match="website_gemini_retained_source_required"):
        _analyze(tmp_path, monkeypatch, video, digest)
