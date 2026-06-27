"""Hermetic fail-closed tests for blueprint_pipeline.splat_scene_render.

The happy path requires node + Spark + ffmpeg and is exercised by the end-to-end
integration run; these tests pin the fail-closed contract (no fabricated passes).
"""
from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.splat_scene_render import (
    RENDERED_BY,
    _encode_mp4,
    render_splat_scene,
)


def test_blocked_when_source_missing(tmp_path: Path) -> None:
    m = render_splat_scene(tmp_path / "nope.ply", tmp_path / "out")
    assert m["status"] == "blocked"
    assert "splat_source_missing_or_unsupported" in m["blockers"]
    assert m["rendered_by"] == RENDERED_BY
    assert m["proof_boundary"]["rendered_by_isaac_rtx"] is False


def test_blocked_when_unsupported_suffix(tmp_path: Path) -> None:
    src = tmp_path / "scene.txt"
    src.write_text("not a splat")
    m = render_splat_scene(src, tmp_path / "out")
    assert m["status"] == "blocked"
    assert "splat_source_missing_or_unsupported" in m["blockers"]


def test_blocked_when_cli_missing(tmp_path: Path) -> None:
    # valid-looking .ply source but repo_root has no splat-transform CLI installed
    src = tmp_path / "scene.ply"
    src.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    m = render_splat_scene(src, tmp_path / "out", repo_root=tmp_path)
    assert m["status"] == "blocked"
    assert "splat_transform_cli_unavailable" in m["blockers"]
    assert m["proof_boundary"]["captured_scene_displayed"] is False


def test_encode_mp4_no_frames(tmp_path: Path) -> None:
    result = _encode_mp4([], tmp_path / "out.mp4")
    assert result["status"] == "blocked"
    assert result["blockers"][0] in {"no_frames_to_encode", "ffmpeg_unavailable"}
