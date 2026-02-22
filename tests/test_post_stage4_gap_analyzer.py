"""Tests for post_stage4_gap_analyzer.py."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure scripts/ is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from post_stage4_gap_analyzer import (
    _angle_between_deg,
    _load_colmap_images_bin,
    _load_colmap_images_txt,
    _load_poses_from_jsonl,
    _qvec_to_rotmat,
    _rotate_yaw_deg,
    _view_dir_from_qvec,
    analyze_gap_observability,
    compute_hole_mask,
    generate_void_filling_candidates,
    rank_candidate_views,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_colmap_images_bin(poses: list[tuple[str, list[float], list[float]]]) -> bytes:
    """Build a minimal COLMAP images.bin from a list of (name, qvec, tvec)."""
    buf = struct.pack("<Q", len(poses))
    for idx, (name, qvec, tvec) in enumerate(poses):
        buf += struct.pack("<I", idx + 1)  # image_id
        buf += struct.pack("<dddd", *qvec)
        buf += struct.pack("<ddd", *tvec)
        buf += struct.pack("<I", 1)  # camera_id
        buf += name.encode("utf-8") + b"\x00"
        buf += struct.pack("<Q", 0)  # num_points2D = 0
    return buf


def _identity_qvec() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Test: COLMAP images.bin reader
# ---------------------------------------------------------------------------

class TestLoadColmapImagesBin:
    def test_reads_valid_binary(self, tmp_path: Path) -> None:
        poses = [
            ("frame_0001.jpg", [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ("frame_0002.jpg", [0.707, 0.707, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ]
        bin_path = tmp_path / "images.bin"
        bin_path.write_bytes(_make_colmap_images_bin(poses))

        result = _load_colmap_images_bin(bin_path)
        assert len(result) == 2
        assert "frame_0001.jpg" in result
        assert "frame_0002.jpg" in result
        assert result["frame_0001.jpg"]["qvec"] == [1.0, 0.0, 0.0, 0.0]
        assert result["frame_0002.jpg"]["tvec"] == [1.0, 0.0, 0.0]

    def test_missing_file(self, tmp_path: Path) -> None:
        result = _load_colmap_images_bin(tmp_path / "nonexistent.bin")
        assert result == {}

    def test_empty_file(self, tmp_path: Path) -> None:
        bin_path = tmp_path / "images.bin"
        bin_path.write_bytes(b"")
        result = _load_colmap_images_bin(bin_path)
        assert result == {}

    def test_truncated_file(self, tmp_path: Path) -> None:
        bin_path = tmp_path / "images.bin"
        bin_path.write_bytes(struct.pack("<Q", 1) + b"\x00" * 10)
        result = _load_colmap_images_bin(bin_path)
        assert result == {}

    def test_zero_images(self, tmp_path: Path) -> None:
        bin_path = tmp_path / "images.bin"
        bin_path.write_bytes(struct.pack("<Q", 0))
        result = _load_colmap_images_bin(bin_path)
        assert result == {}

    def test_with_2d_points(self, tmp_path: Path) -> None:
        """Images with 2D point observations should be skipped correctly."""
        buf = struct.pack("<Q", 1)  # 1 image
        buf += struct.pack("<I", 1)  # image_id
        buf += struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0)  # qvec
        buf += struct.pack("<ddd", 0.0, 0.0, 0.0)  # tvec
        buf += struct.pack("<I", 1)  # camera_id
        buf += b"test.jpg\x00"  # name
        buf += struct.pack("<Q", 3)  # 3 points2D
        # Each point2D: x(double) + y(double) + point3D_id(int64) = 24 bytes
        for _ in range(3):
            buf += struct.pack("<ddq", 100.0, 200.0, -1)
        bin_path = tmp_path / "images.bin"
        bin_path.write_bytes(buf)
        result = _load_colmap_images_bin(bin_path)
        assert len(result) == 1
        assert "test.jpg" in result


# ---------------------------------------------------------------------------
# Test: COLMAP images.txt reader
# ---------------------------------------------------------------------------

class TestLoadColmapImagesTxt:
    def test_reads_valid_text(self, tmp_path: Path) -> None:
        txt_path = tmp_path / "images.txt"
        txt_path.write_text(
            "# image_id qw qx qy qz tx ty tz camera_id name\n"
            "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_0001.jpg\n"
            "\n"
            "2 0.707 0.707 0.0 0.0 1.0 0.0 0.0 1 frame_0002.jpg\n"
            "\n"
        )
        result = _load_colmap_images_txt(txt_path)
        assert len(result) == 2
        assert result["frame_0001.jpg"]["qvec"] == [1.0, 0.0, 0.0, 0.0]

    def test_missing_file(self, tmp_path: Path) -> None:
        result = _load_colmap_images_txt(tmp_path / "no.txt")
        assert result == {}


# ---------------------------------------------------------------------------
# Test: Poses JSONL reader
# ---------------------------------------------------------------------------

class TestLoadPosesFromJsonl:
    def test_reads_valid_jsonl(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "poses.jsonl"
        jsonl_path.write_text(
            json.dumps({"image": "frame_01.jpg", "qvec": [1, 0, 0, 0], "tvec": [0, 0, 0]}) + "\n"
            + json.dumps({"name": "frame_02.jpg", "qvec": [0.5, 0.5, 0.5, 0.5], "tvec": [1, 2, 3]}) + "\n"
        )
        result = _load_poses_from_jsonl(jsonl_path)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Test: Hole mask computation
# ---------------------------------------------------------------------------

class TestComputeHoleMask:
    def test_dark_region_detected(self) -> None:
        # Black image = 100% hole
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = compute_hole_mask(rgb)
        assert mask.all()

    def test_bright_region_not_hole(self) -> None:
        # Bright white image = 0% hole
        rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = compute_hole_mask(rgb)
        assert not mask.any()

    def test_alpha_zero_is_hole(self) -> None:
        rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
        alpha = np.zeros((100, 100), dtype=np.uint8)
        mask = compute_hole_mask(rgb, alpha=alpha)
        assert mask.all()

    def test_mixed_image(self) -> None:
        rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
        rgb[:50, :, :] = 0  # top half dark
        mask = compute_hole_mask(rgb)
        assert mask[:50, :].all()
        assert not mask[50:, :].any()


# ---------------------------------------------------------------------------
# Test: Candidate ranking
# ---------------------------------------------------------------------------

class TestRankCandidateViews:
    def test_empty(self) -> None:
        result = rank_candidate_views([], max_candidates=10, min_parallax_deg=5.0)
        assert result == []

    def test_respects_max_candidates(self) -> None:
        candidates = [
            {
                "score": float(i),
                "hole_ratio": 0.5,
                "cluster_count": 1,
                "sharpness": 100.0,
                "parallax_to_nearest_captured_deg": 15.0,
                "view_dir": [math.sin(i * 0.5), 0.0, math.cos(i * 0.5)],
            }
            for i in range(20)
        ]
        result = rank_candidate_views(candidates, max_candidates=5, min_parallax_deg=0.0)
        assert len(result) <= 5

    def test_filters_by_parallax(self) -> None:
        candidates = [
            {
                "score": 10.0,
                "hole_ratio": 0.5,
                "cluster_count": 1,
                "sharpness": 100.0,
                "parallax_to_nearest_captured_deg": 3.0,  # Below threshold
                "view_dir": [0.0, 0.0, 1.0],
            },
        ]
        result = rank_candidate_views(candidates, max_candidates=10, min_parallax_deg=5.0)
        assert result == []


# ---------------------------------------------------------------------------
# Test: Geometry helpers
# ---------------------------------------------------------------------------

class TestGeometryHelpers:
    def test_identity_qvec_rotmat(self) -> None:
        rot = _qvec_to_rotmat([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-10)

    def test_view_dir_identity(self) -> None:
        """Identity quaternion should give forward = [0,0,1]."""
        vd = _view_dir_from_qvec([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(vd, [0.0, 0.0, 1.0], atol=1e-10)

    def test_rotate_yaw_zero(self) -> None:
        vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        result = _rotate_yaw_deg(vec, 0.0)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_rotate_yaw_90(self) -> None:
        vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        result = _rotate_yaw_deg(vec, 90.0)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)

    def test_angle_between_parallel(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        assert _angle_between_deg(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_angle_between_orthogonal(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert _angle_between_deg(a, b) == pytest.approx(90.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: analyze_gap_observability E2E
# ---------------------------------------------------------------------------

class TestAnalyzeGapObservability:
    def test_e2e_with_renders(self, tmp_path: Path) -> None:
        """End-to-end: create synthetic renders and run analysis."""
        PILImage = pytest.importorskip("PIL.Image")
        renders_dir = tmp_path / "renders"
        renders_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a few synthetic render images (half black, half white).
        for i in range(5):
            img = np.full((64, 64, 3), 200, dtype=np.uint8)
            img[:32, :, :] = 0  # top half dark
            PILImage.fromarray(img).save(renders_dir / f"frame_{i:04d}.png")

        report = analyze_gap_observability(
            renders_dir=renders_dir,
            output_dir=output_dir,
            max_candidate_views=10,
            min_parallax_deg=5.0,
        )

        assert report["input_render_count"] == 5
        assert report["global_hole_pixel_ratio"] > 0.0
        assert (output_dir / "gap_analysis_report.json").is_file()
        assert (output_dir / "gap_candidate_views.jsonl").is_file()

    def test_e2e_with_colmap_bin_poses(self, tmp_path: Path) -> None:
        """Verifies binary COLMAP pose loading integrates with analysis."""
        PILImage = pytest.importorskip("PIL.Image")
        renders_dir = tmp_path / "renders"
        renders_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create render image with matching COLMAP name.
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        PILImage.fromarray(img).save(renders_dir / "frame_0001.png")

        # Create COLMAP binary with a matching pose.
        poses = [("frame_0001.png", [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0])]
        bin_path = tmp_path / "images.bin"
        bin_path.write_bytes(_make_colmap_images_bin(poses))

        report = analyze_gap_observability(
            renders_dir=renders_dir,
            output_dir=output_dir,
            max_candidate_views=10,
            min_parallax_deg=5.0,
            colmap_images_bin=bin_path,
        )

        assert report["input_render_count"] == 1
        # Global hole ratio should be ~1.0 (all black image).
        assert report["global_hole_pixel_ratio"] > 0.9

    def test_resolves_numeric_render_name_with_colmap_index_mapping(self, tmp_path: Path) -> None:
        """Render 00000.png should map to COLMAP image_id order when names differ."""
        PILImage = pytest.importorskip("PIL.Image")
        renders_dir = tmp_path / "renders"
        renders_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        PILImage.fromarray(img).save(renders_dir / "00000.png")

        # COLMAP name does not match render name, but index 0 should match.
        poses = [("frame_0042.jpg", [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0])]
        bin_path = tmp_path / "images.bin"
        bin_path.write_bytes(_make_colmap_images_bin(poses))

        report = analyze_gap_observability(
            renders_dir=renders_dir,
            output_dir=output_dir,
            max_candidate_views=10,
            min_parallax_deg=5.0,
            colmap_images_bin=bin_path,
        )

        assert report["pose_mapping_mode"] == "name_or_colmap_index"
        assert report["pose_match_count"] == 1
        assert report["pose_fallback_count"] == 0
        assert report["pose_index_match_count"] == 1


# ---------------------------------------------------------------------------
# Test: generate_void_filling_candidates
# ---------------------------------------------------------------------------

class TestGenerateVoidFillingCandidates:
    def _horizontal_ring_poses(self) -> dict[str, dict[str, Any]]:
        """Create poses only in the horizontal ring — leaves poles uncovered."""
        poses: dict[str, dict[str, Any]] = {}
        for i in range(12):
            phi = i * (2.0 * math.pi / 12)
            # Camera at radius 3 on horizontal ring, looking inward
            eye = np.array([3.0 * math.sin(phi), 0.0, 3.0 * math.cos(phi)])
            forward = -eye / np.linalg.norm(eye)
            # Simple qvec — identity is fine; coverage is computed from
            # direction of (eye - scene_center), not from qvec.
            poses[f"frame_{i:04d}.jpg"] = {
                "qvec": [1.0, 0.0, 0.0, 0.0],
                "tvec": list(-eye),  # tvec = -R @ center, with R=I
            }
        return poses

    def test_generates_candidates_near_poles(self) -> None:
        """Virtual cameras should be generated near floor/ceiling (poles)."""
        poses = self._horizontal_ring_poses()
        candidates = generate_void_filling_candidates(
            scene_center=np.zeros(3),
            scene_radius=2.0,
            existing_poses=poses,
            max_candidates=48,
        )
        assert len(candidates) > 0
        # Check that at least one candidate has a high or low elevation
        elevations = []
        for c in candidates:
            eye = np.array(c["camera_center"])
            r = np.linalg.norm(eye)
            if r > 1e-8:
                y_norm = abs(eye[1]) / r
                elevations.append(y_norm)
        # At least one candidate should point toward pole (|y/r| > 0.7)
        assert any(e > 0.7 for e in elevations), (
            f"No near-pole candidates found; max |elevation|={max(elevations):.2f}"
        )

    def test_well_covered_bins_skipped_not_blocking(self) -> None:
        """Well-covered bins (score<=0.25) should be skipped, not block others."""
        # Create dense coverage in some directions but leave gaps
        poses: dict[str, dict[str, Any]] = {}
        # 20 cameras all pointing roughly the same direction
        for i in range(20):
            angle = i * 0.01  # tiny spread
            eye = np.array([3.0 * math.sin(angle), 0.0, 3.0 * math.cos(angle)])
            poses[f"dense_{i:04d}.jpg"] = {
                "qvec": [1.0, 0.0, 0.0, 0.0],
                "tvec": list(-eye),
            }
        candidates = generate_void_filling_candidates(
            scene_center=np.zeros(3),
            scene_radius=2.0,
            existing_poses=poses,
            max_candidates=48,
        )
        # Should still generate candidates for the uncovered directions
        assert len(candidates) > 0

    def test_exclude_poles_removes_near_pole_candidates(self) -> None:
        poses = self._horizontal_ring_poses()
        candidates = generate_void_filling_candidates(
            scene_center=np.zeros(3),
            scene_radius=2.0,
            existing_poses=poses,
            max_candidates=96,
            exclude_poles=True,
            pole_exclusion_fraction=0.15,
        )
        assert len(candidates) > 0
        elevations = []
        for c in candidates:
            eye = np.array(c["camera_center"])
            r = np.linalg.norm(eye)
            if r > 1e-8:
                elevations.append(abs(eye[1]) / r)
        assert max(elevations) < 0.85

    def test_empty_poses_returns_empty(self) -> None:
        candidates = generate_void_filling_candidates(
            scene_center=np.zeros(3),
            scene_radius=2.0,
            existing_poses={},
            max_candidates=48,
        )
        assert candidates == []
