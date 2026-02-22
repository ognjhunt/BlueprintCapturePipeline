"""Tests for post_stage4_distill.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from post_stage4_distill import (
    _append_virtual_from_accepted_rows,
    _build_render_index_to_frame_map,
    _copy_matching_repaired_views,
    _ensure_sparse_text_model,
    _load_jsonl,
    _regenerate_sparse_bin_model,
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

    def test_skips_virtual_rows_for_overlay_replacement(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "frame_01.png").write_text("original")

        repaired_dir = tmp_path / "repaired"
        repaired_dir.mkdir()
        repaired_virtual = repaired_dir / "virtual_fixed.png"
        repaired_virtual.write_text("fixed")

        jsonl_path = tmp_path / "accepted.jsonl"
        jsonl_path.write_text(
            json.dumps(
                {
                    "is_virtual": True,
                    "source_image": "frame_01.png",
                    "repaired_image": str(repaired_virtual),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        count, _ = _copy_matching_repaired_views(
            undistorted_images_dir=images_dir,
            repaired_views_dir=repaired_dir,
            accepted_views_jsonl=jsonl_path,
        )
        assert count == 0
        assert (images_dir / "frame_01.png").read_text() == "original"


def test_append_virtual_from_accepted_rows_uses_repaired_and_camera_id(tmp_path: Path) -> None:
    sparse_dir = tmp_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_txt = sparse_dir / "images.txt"
    images_txt.write_text("# header\n", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    repaired = tmp_path / "repaired_virtual.png"
    repaired.write_bytes(b"PNGDATA")
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text(
        json.dumps(
            {
                "is_virtual": True,
                "repaired_image": str(repaired),
                "qvec": [1.0, 0.0, 0.0, 0.0],
                "tvec": [0.0, 1.0, 2.0],
                "camera_id": 9,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    appended, missing = _append_virtual_from_accepted_rows(
        sparse_dir=sparse_dir,
        accepted_views_jsonl=accepted,
        images_dir=images_dir,
        starting_id=10,
        default_camera_id=3,
    )
    assert appended == 1
    assert missing == 0
    assert (images_dir / "virtual_00000.png").is_file()
    lines = [line for line in images_txt.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert len(lines) == 1
    assert lines[0].split()[8] == "9"


def test_sparse_model_text_and_bin_conversion_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sparse_dir = tmp_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for name in ("images.bin", "cameras.bin", "points3D.bin"):
        (sparse_dir / name).write_bytes(b"\x00")

    def _fake_converter(*, input_path: Path, output_path: Path, output_type: str):  # type: ignore[no-untyped-def]
        del input_path
        output_path.mkdir(parents=True, exist_ok=True)
        if output_type == "TXT":
            (output_path / "images.txt").write_text("#\n", encoding="utf-8")
            (output_path / "cameras.txt").write_text("7 PINHOLE 8 8 4 4 4 4\n", encoding="utf-8")
            (output_path / "points3D.txt").write_text("#\n", encoding="utf-8")
            return True, ""
        if output_type == "BIN":
            (output_path / "images.bin").write_bytes(b"\x01")
            (output_path / "cameras.bin").write_bytes(b"\x02")
            (output_path / "points3D.bin").write_bytes(b"\x03")
            return True, ""
        return False, "unsupported"

    monkeypatch.setattr("post_stage4_distill._run_colmap_model_converter", _fake_converter)
    ok_txt, reason_txt = _ensure_sparse_text_model(sparse_dir)
    assert ok_txt is True
    assert reason_txt == ""
    assert (sparse_dir / "images.txt").is_file()
    assert (sparse_dir / "cameras.txt").is_file()
    assert (sparse_dir / "points3D.txt").is_file()

    ok_bin, reason_bin = _regenerate_sparse_bin_model(sparse_dir)
    assert ok_bin is True
    assert reason_bin == ""
    assert (sparse_dir / "images.bin").read_bytes() == b"\x01"
    assert (sparse_dir / "cameras.bin").read_bytes() == b"\x02"
    assert (sparse_dir / "points3D.bin").read_bytes() == b"\x03"


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

    def test_virtual_append_failure_marks_non_ok_and_machine_fields(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        undistorted_dir = tmp_path / "undistorted"
        images_dir = undistorted_dir / "images"
        sparse_dir = undistorted_dir / "sparse" / "0"
        images_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)
        (sparse_dir / "images.txt").write_text("#\n", encoding="utf-8")
        (sparse_dir / "cameras.txt").write_text("2 PINHOLE 8 8 4 4 4 4\n", encoding="utf-8")
        (sparse_dir / "points3D.txt").write_text("#\n", encoding="utf-8")

        base_usdz = tmp_path / "base.usdz"
        base_ply = tmp_path / "base.ply"
        base_usdz.write_bytes(b"USDZ_DATA")
        base_ply.write_bytes(b"PLY_DATA")

        repaired_virtual = tmp_path / "repaired_virtual.png"
        repaired_virtual.write_bytes(b"IMG")
        accepted_jsonl = tmp_path / "accepted.jsonl"
        accepted_jsonl.write_text(
            json.dumps(
                {
                    "is_virtual": True,
                    "repaired_image": str(repaired_virtual),
                    # Missing qvec/tvec on purpose -> append failure
                }
            )
            + "\n",
            encoding="utf-8",
        )

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

        assert report["status"] == "fallback_baseline_copy_virtual_append_failed"
        assert report["distill_ok"] is False
        assert report["virtual_appended_count"] == 0
        assert report["virtual_append_failed_reason"] == "virtual_accepted_rows_missing_required_fields"

    def test_bin_only_sparse_model_virtual_append_works(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        undistorted_dir = tmp_path / "undistorted"
        images_dir = undistorted_dir / "images"
        sparse_dir = undistorted_dir / "sparse" / "0"
        images_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)
        (images_dir / "frame_00001.jpg").write_bytes(b"ORIGINAL")

        # BIN-only input model.
        for name in ("images.bin", "cameras.bin", "points3D.bin"):
            (sparse_dir / name).write_bytes(b"\x00")

        base_usdz = tmp_path / "base.usdz"
        base_ply = tmp_path / "base.ply"
        base_usdz.write_bytes(b"USDZ_DATA")
        base_ply.write_bytes(b"PLY_DATA")

        repaired_virtual = tmp_path / "repaired_virtual.png"
        repaired_virtual.write_bytes(b"REPAIRED")
        accepted_jsonl = tmp_path / "accepted.jsonl"
        accepted_jsonl.write_text(
            json.dumps(
                {
                    "is_virtual": True,
                    "repaired_image": str(repaired_virtual),
                    "qvec": [1.0, 0.0, 0.0, 0.0],
                    "tvec": [0.0, 0.0, 0.0],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        def _fake_converter(*, input_path: Path, output_path: Path, output_type: str):  # type: ignore[no-untyped-def]
            del input_path
            output_path.mkdir(parents=True, exist_ok=True)
            if output_type == "TXT":
                (output_path / "images.txt").write_text("#\n", encoding="utf-8")
                (output_path / "cameras.txt").write_text("7 PINHOLE 8 8 4 4 4 4\n", encoding="utf-8")
                (output_path / "points3D.txt").write_text("#\n", encoding="utf-8")
                return True, ""
            (output_path / "images.bin").write_bytes(b"\x09")
            (output_path / "cameras.bin").write_bytes(b"\x08")
            (output_path / "points3D.bin").write_bytes(b"\x07")
            return True, ""

        fake_result_dir = tmp_path / "distill_result"
        fake_result_dir.mkdir(parents=True)
        (fake_result_dir / "export_last.usdz").write_bytes(b"U")
        (fake_result_dir / "export_last.ply").write_bytes(b"P")
        (fake_result_dir / "metrics.json").write_text(json.dumps({"mean_psnr": 25.0}), encoding="utf-8")

        monkeypatch.setattr("post_stage4_distill._run_colmap_model_converter", _fake_converter)
        monkeypatch.setattr(
            "post_stage4_distill._run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr("post_stage4_distill._find_latest_result", lambda *_args, **_kwargs: fake_result_dir)

        report = run_post_stage4_distill(
            output_dir=output_dir,
            undistorted_dir=undistorted_dir,
            base_usdz=base_usdz,
            base_ply=base_ply,
            base_ingp=None,
            accepted_views_jsonl=accepted_jsonl,
            repaired_views_dir=tmp_path / "repaired",
            distill_iters=10,
            max_n_gaussians=0,
            time_budget_min=5,
            threedgrut_python="python3.11",
            threedgrut_dir=tmp_path,
        )

        assert report["status"] == "ok"
        assert report["distill_ok"] is True
        assert report["virtual_appended_count"] == 1
        assert report["resume_disabled_reason"] == "virtual_views_appended_camera_count_changed"

        sparse_copy = Path(report["work_dir"]) / "undistorted_refine" / "sparse" / "0"
        images_txt = sparse_copy / "images.txt"
        lines = [line for line in images_txt.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
        assert lines, "expected appended virtual image line"
        assert lines[-1].split()[8] == "7"
        assert (sparse_copy / "images.bin").read_bytes() == b"\x09"
