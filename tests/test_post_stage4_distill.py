"""Tests for post_stage4_distill.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from post_stage4_distill import (
    _build_render_index_to_frame_map,
    _copy_matching_repaired_views,
    _load_jsonl,
    run_post_stage4_distill,
)


# ---------------------------------------------------------------------------
# Test: _load_jsonl
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_reads_valid_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"a": 1}) + "\n"
            + json.dumps({"b": 2}) + "\n"
        )
        rows = _load_jsonl(path)
        assert len(rows) == 2
        assert rows[0]["a"] == 1
        assert rows[1]["b"] == 2

    def test_missing_file(self, tmp_path: Path) -> None:
        assert _load_jsonl(tmp_path / "nope.jsonl") == []

    def test_skips_invalid_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text("not json\n" + json.dumps({"ok": True}) + "\n")
        rows = _load_jsonl(path)
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Test: _copy_matching_repaired_views
# ---------------------------------------------------------------------------

class TestBuildRenderIndexToFrameMap:
    def test_maps_render_indices_to_frame_names(self, tmp_path: Path) -> None:
        """3DGRUT renders 00000.png..00002.png → frame_00001.jpg, frame_00003.jpg, frame_00005.jpg."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        # Gaps in numbering (frame_00002 and frame_00004 missing — dropped by SfM)
        (images_dir / "frame_00001.jpg").write_text("a")
        (images_dir / "frame_00003.jpg").write_text("b")
        (images_dir / "frame_00005.jpg").write_text("c")

        mapping = _build_render_index_to_frame_map(images_dir)
        assert mapping == {
            "00000.png": "frame_00001.jpg",
            "00001.png": "frame_00003.jpg",
            "00002.png": "frame_00005.jpg",
        }

    def test_empty_directory(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        assert _build_render_index_to_frame_map(images_dir) == {}

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "frame_00001.jpg").write_text("a")
        (images_dir / "cameras.txt").write_text("not an image")
        (images_dir / "points3D.bin").write_bytes(b"\x00")

        mapping = _build_render_index_to_frame_map(images_dir)
        assert mapping == {"00000.png": "frame_00001.jpg"}


class TestCopyMatchingRepairedViews:
    def test_copies_matching_images(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "frame_01.png").write_text("original")
        (images_dir / "frame_02.png").write_text("original")

        repaired_dir = tmp_path / "repaired"
        repaired_dir.mkdir()
        (repaired_dir / "frame_01.png").write_text("fixed")

        jsonl_path = tmp_path / "accepted.jsonl"
        jsonl_path.write_text(
            json.dumps({
                "source_image": "frame_01.png",
                "repaired_image": str(repaired_dir / "frame_01.png"),
            }) + "\n"
        )

        count, paths = _copy_matching_repaired_views(
            undistorted_images_dir=images_dir,
            repaired_views_dir=repaired_dir,
            accepted_views_jsonl=jsonl_path,
        )
        assert count == 1
        assert (images_dir / "frame_01.png").read_text() == "fixed"
        assert (images_dir / "frame_02.png").read_text() == "original"

    def test_translates_render_index_to_frame_name(self, tmp_path: Path) -> None:
        """source_image='00000.png' should map to frame_00001.jpg via render index."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "frame_00001.jpg").write_text("original_1")
        (images_dir / "frame_00003.jpg").write_text("original_3")

        repaired_dir = tmp_path / "repaired"
        repaired_dir.mkdir()
        (repaired_dir / "00000.png").write_text("fixed_0")

        jsonl_path = tmp_path / "accepted.jsonl"
        jsonl_path.write_text(
            json.dumps({
                "source_image": "00000.png",
                "repaired_image": str(repaired_dir / "00000.png"),
            }) + "\n"
        )

        count, paths = _copy_matching_repaired_views(
            undistorted_images_dir=images_dir,
            repaired_views_dir=repaired_dir,
            accepted_views_jsonl=jsonl_path,
        )
        assert count == 1
        # 00000.png → frame_00001.jpg (first in sorted order)
        assert (images_dir / "frame_00001.jpg").read_text() == "fixed_0"
        assert (images_dir / "frame_00003.jpg").read_text() == "original_3"

    def test_deduplicates_same_source(self, tmp_path: Path) -> None:
        """Multiple accepted views from the same source should only copy once."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "frame_00001.jpg").write_text("original")

        repaired_dir = tmp_path / "repaired"
        repaired_dir.mkdir()
        (repaired_dir / "00000.png").write_text("fixed")

        jsonl_path = tmp_path / "accepted.jsonl"
        jsonl_path.write_text(
            json.dumps({"source_image": "00000.png", "repaired_image": str(repaired_dir / "00000.png")}) + "\n"
            + json.dumps({"source_image": "00000.png", "repaired_image": str(repaired_dir / "00000.png")}) + "\n"
        )

        count, _ = _copy_matching_repaired_views(
            undistorted_images_dir=images_dir,
            repaired_views_dir=repaired_dir,
            accepted_views_jsonl=jsonl_path,
        )
        assert count == 1

    def test_no_matching_returns_zero(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        jsonl_path = tmp_path / "accepted.jsonl"
        jsonl_path.write_text("")

        count, paths = _copy_matching_repaired_views(
            undistorted_images_dir=images_dir,
            repaired_views_dir=tmp_path / "repaired",
            accepted_views_jsonl=jsonl_path,
        )
        assert count == 0
        assert paths == []


# ---------------------------------------------------------------------------
# Test: run_post_stage4_distill
# ---------------------------------------------------------------------------

class TestRunPostStage4Distill:
    def test_skips_when_no_accepted_views(self, tmp_path: Path) -> None:
        """When no repaired views match, distill copies baseline."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        undistorted_dir = tmp_path / "undistorted"
        undistorted_dir.mkdir()
        (undistorted_dir / "images").mkdir()
        (undistorted_dir / "sparse").mkdir(parents=True)

        base_usdz = tmp_path / "base.usdz"
        base_ply = tmp_path / "base.ply"
        base_usdz.write_bytes(b"USDZ_DATA")
        base_ply.write_bytes(b"PLY_DATA")

        accepted_jsonl = tmp_path / "accepted.jsonl"
        accepted_jsonl.write_text("")

        report = run_post_stage4_distill(
            output_dir=output_dir,
            undistorted_dir=undistorted_dir,
            base_usdz=base_usdz,
            base_ply=base_ply,
            base_ingp=None,
            accepted_views_jsonl=accepted_jsonl,
            repaired_views_dir=tmp_path / "repaired",
            distill_iters=100,
            max_n_gaussians=0,
            time_budget_min=5,
            threedgrut_python="python3.11",
            threedgrut_dir=Path("/opt/3dgrut"),
        )

        assert report["status"] == "skipped_no_matching_repaired_views"
        assert (output_dir / "export_last_refined.usdz").read_bytes() == b"USDZ_DATA"
        assert (output_dir / "export_last_refined.ply").read_bytes() == b"PLY_DATA"
