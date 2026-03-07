"""Tests for post_stage4_virtual_render.py."""

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

from post_stage4_virtual_render import (
    _estimate_hole_ratio,
    _is_valid_camera_dims,
    build_virtual_colmap_dataset,
    render_and_collect_virtual_views,
)


def test_estimate_hole_ratio_deterministic_on_synthetic_images(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    PILImage = pytest.importorskip("PIL.Image")
    black = tmp_path / "black.png"
    white = tmp_path / "white.png"
    PILImage.fromarray((0 * np.ones((8, 8, 3))).astype("uint8")).save(black)
    PILImage.fromarray((255 * np.ones((8, 8, 3))).astype("uint8")).save(white)
    assert _estimate_hole_ratio(black) == pytest.approx(1.0)
    assert _estimate_hole_ratio(white) == pytest.approx(0.0)


def test_mapping_includes_absolute_render_path_pose_and_hole_ratio(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    PILImage = pytest.importorskip("PIL.Image")

    reference_sparse = tmp_path / "sparse" / "0"
    reference_sparse.mkdir(parents=True, exist_ok=True)
    (reference_sparse / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "7 PINHOLE 8 8 4.0 4.0 4.0 4.0\n",
        encoding="utf-8",
    )

    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "id": "cand_1",
                "is_virtual": True,
                "source_image": "00000.png",
                "qvec": [1.0, 0.0, 0.0, 0.0],
                "tvec": [0.0, 0.0, 0.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    renders_dir = work_dir / "fake_renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    img = np.full((8, 8, 3), 255, dtype=np.uint8)
    img[:4, :, :] = 0
    PILImage.fromarray(img).save(renders_dir / "00000.png")

    def _fake_render(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return True, renders_dir, ""

    with mock.patch("post_stage4_virtual_render.render_virtual_views", side_effect=_fake_render):
        report = render_and_collect_virtual_views(
            candidates_jsonl=candidates,
            checkpoint_path=tmp_path / "ckpt_last.pt",
            reference_sparse_dir=reference_sparse,
            work_dir=work_dir,
            threedgrut_python=sys.executable,
            threedgrut_dir=tmp_path,
        )

    assert report["status"] == "ok"
    mapping_path = Path(report["mapping_path"])
    rows = [json.loads(line) for line in mapping_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == "cand_1"
    assert row["render_name"] == "00000.png"
    assert row["is_virtual"] is True
    assert row["camera_id"] == 7
    assert row["qvec"] == [1.0, 0.0, 0.0, 0.0]
    assert row["tvec"] == [0.0, 0.0, 0.0]
    assert row["render_exists"] is True
    assert Path(row["render_image"]).is_absolute()
    assert float(row["predicted_hole_ratio"]) == pytest.approx(0.5, abs=1e-6)


def test_camera_dimension_validation_bounds() -> None:
    assert _is_valid_camera_dims(8, 8) is True
    assert _is_valid_camera_dims(0, 8) is False
    assert _is_valid_camera_dims(8, 0) is False
    assert _is_valid_camera_dims(9000, 8) is False
    assert _is_valid_camera_dims(8, 9000) is False


def test_build_virtual_colmap_dataset_rejects_oversized_camera(tmp_path: Path) -> None:
    reference_camera = {
        "camera_id": 1,
        "model": "PINHOLE",
        "width": 100000,
        "height": 100000,
        "params": [1.0, 1.0, 1.0, 1.0],
    }
    candidates = [
        {
            "qvec": [1.0, 0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 0.0],
        }
    ]

    with pytest.raises(ValueError, match="Invalid reference camera dimensions"):
        build_virtual_colmap_dataset(tmp_path / "virtual_dataset", candidates, reference_camera)


def test_render_collect_rejects_invalid_reference_camera_dimensions(tmp_path: Path) -> None:
    reference_sparse = tmp_path / "sparse" / "0"
    reference_sparse.mkdir(parents=True, exist_ok=True)
    (reference_sparse / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "7 PINHOLE 100000 100000 4.0 4.0 4.0 4.0\n",
        encoding="utf-8",
    )

    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "id": "cand_1",
                "is_virtual": True,
                "source_image": "00000.png",
                "qvec": [1.0, 0.0, 0.0, 0.0],
                "tvec": [0.0, 0.0, 0.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = render_and_collect_virtual_views(
        candidates_jsonl=candidates,
        checkpoint_path=tmp_path / "ckpt_last.pt",
        reference_sparse_dir=reference_sparse,
        work_dir=tmp_path / "work",
        threedgrut_python=sys.executable,
        threedgrut_dir=tmp_path,
    )

    assert report["status"] == "error_invalid_reference_camera_dimensions"
    assert report["rendered_count"] == 0
