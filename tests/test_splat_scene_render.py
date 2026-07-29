"""Hermetic fail-closed tests for blueprint_pipeline.splat_scene_render.

The happy path requires node + Spark + ffmpeg and is exercised by the end-to-end
integration run; these tests pin the fail-closed contract (no fabricated passes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.splat_scene_render import (
    RENDERED_BY,
    _decimate_to_standard_ply,
    _encode_mp4,
    render_splat_scene,
)


def test_valid_standard_ply_needs_no_decoder_when_decimation_disabled(
    tmp_path: Path,
) -> None:
    count = 4
    source = write_standard_3dgs_ply(
        SplatData(
            count=count,
            xyz=np.zeros((count, 3), dtype=np.float32),
            opacity=np.zeros(count, dtype=np.float32),
            f_dc=np.zeros((count, 3), dtype=np.float32),
            scales=np.zeros((count, 3), dtype=np.float32),
            quats=np.zeros((count, 4), dtype=np.float32),
            properties=(),
        ),
        tmp_path / "source.ply",
    )
    result = _decimate_to_standard_ply(
        source,
        tmp_path / "out" / "scene_standard.ply",
        0,
        repo_root=tmp_path,
        node="node",
        timeout=10,
    )
    assert result["status"] == "completed"
    assert result["decoder"] == "validated_standard_3dgs_copy"
    assert result["vertex_count"] == count


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


def test_blocked_when_graphics_backend_is_unknown(tmp_path: Path) -> None:
    src = tmp_path / "scene.ply"
    src.write_bytes(b"placeholder")
    m = render_splat_scene(src, tmp_path / "out", graphics_backend="cloud_magic")
    assert m["status"] == "blocked"
    assert m["blockers"] == ["unsupported_graphics_backend"]


def test_blocked_when_cli_missing(tmp_path: Path) -> None:
    # valid-looking .ply source but repo_root has no splat-transform CLI installed
    src = tmp_path / "scene.ply"
    src.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    m = render_splat_scene(src, tmp_path / "out", repo_root=tmp_path)
    assert m["status"] == "blocked"
    assert "splat_transform_cli_unavailable" in m["blockers"]
    assert m["proof_boundary"]["captured_scene_displayed"] is False


def test_invalid_focus_point_fails_closed_after_decode(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "scene.ply"
    src.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "blueprint_pipeline.splat_scene_render._decimate_to_standard_ply",
        lambda *args, **kwargs: {"status": "completed"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.splat_scene_render.read_standard_3dgs_ply",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.splat_scene_render.analyze_scene",
        lambda *args, **kwargs: type("Geometry", (), {"to_dict": lambda self: {}})(),
    )
    result = render_splat_scene(src, tmp_path / "out", focus_point=[1.0, float("nan"), 2.0])
    assert result["status"] == "blocked"
    assert result["blockers"] == ["invalid_focus_point"]


def test_encode_mp4_no_frames(tmp_path: Path) -> None:
    result = _encode_mp4([], tmp_path / "out.mp4")
    assert result["status"] == "blocked"
    assert result["blockers"][0] in {"no_frames_to_encode", "ffmpeg_unavailable"}
