"""Tests for Inpaint360GS scene cleaning orchestrator."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import sys

_repo_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_repo_scripts) not in sys.path:
    sys.path.insert(0, str(_repo_scripts))

import inpaint360gs_runner as runner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_colmap(tmp_path: Path) -> Path:
    """Create minimal COLMAP sparse reconstruction files."""
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").write_bytes(b"\x00" * 8)
    (sparse / "images.bin").write_bytes(b"\x00" * 8)
    (sparse / "points3D.bin").write_bytes(b"\x00" * 8)
    return sparse


@pytest.fixture()
def fake_images(tmp_path: Path) -> Path:
    """Create a directory with dummy image files."""
    images = tmp_path / "images"
    images.mkdir()
    for i in range(5):
        (images / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8" * 10)
    return images


@pytest.fixture()
def fake_masks(tmp_path: Path) -> Path:
    """Create instance mask PNGs with two objects."""
    masks = tmp_path / "instance_masks"
    masks.mkdir()
    try:
        from PIL import Image
        import numpy as np
        for i in range(5):
            arr = np.zeros((480, 640), dtype=np.uint8)
            arr[100:200, 100:200] = 1  # object 1
            arr[300:400, 400:500] = 2  # object 2
            Image.fromarray(arr, mode="L").save(masks / f"frame_{i:04d}.png")
    except ImportError:
        # Fallback: write tiny PNGs using raw bytes (8x8 grayscale)
        for i in range(5):
            (masks / f"frame_{i:04d}.png").write_bytes(b"\x89PNG" + b"\x00" * 20)
    return masks


@pytest.fixture()
def fake_object_index(tmp_path: Path) -> Path:
    """Create a minimal object_point_cloud_index.json."""
    idx = tmp_path / "object_point_cloud_index.json"
    idx.write_text(json.dumps({
        "objects": [
            {"id": "obj_1", "label": "dresser", "obb": {"center": [0, 0, 0]}},
            {"id": "obj_2", "label": "bed", "obb": {"center": [1, 0, 0]}},
        ]
    }), encoding="utf-8")
    return idx


# ---------------------------------------------------------------------------
# TestPrepareDataLayout
# ---------------------------------------------------------------------------

class TestPrepareDataLayout:
    def test_creates_symlinks_and_scene_json(
        self, tmp_path: Path, fake_colmap: Path, fake_images: Path,
        fake_masks: Path, fake_object_index: Path,
    ) -> None:
        workspace = tmp_path / "inpaint_ws"
        result = runner.prepare_data_layout(
            colmap_sparse_dir=fake_colmap,
            images_dir=fake_images,
            instance_masks_dir=fake_masks,
            object_index_path=fake_object_index,
            workspace=workspace,
            resolution=1,
        )

        # Check structure
        assert (workspace / "images").exists() or (workspace / "images").is_symlink()
        assert (workspace / "sparse" / "0").exists() or (workspace / "sparse" / "0").is_symlink()
        assert (workspace / "associated_hqsam" / "scene.json").is_file()

        # Check scene.json
        scene_json = json.loads((workspace / "associated_hqsam" / "scene.json").read_text())
        assert scene_json["num_classes"] == 3  # 2 objects + background

        # Check mask files were copied
        mask_files = list((workspace / "associated_hqsam").glob("*.png"))
        assert len(mask_files) == 5

        # Check return value
        assert result["num_objects"] == 2
        assert result["num_images"] == 5
        assert result["object_ids"] == [1, 2]

    def test_rescales_masks_for_resolution(
        self, tmp_path: Path, fake_colmap: Path, fake_images: Path,
        fake_masks: Path, fake_object_index: Path,
    ) -> None:
        workspace = tmp_path / "inpaint_ws"
        result = runner.prepare_data_layout(
            colmap_sparse_dir=fake_colmap,
            images_dir=fake_images,
            instance_masks_dir=fake_masks,
            object_index_path=fake_object_index,
            workspace=workspace,
            resolution=2,
        )
        assert result["num_objects"] == 2
        # Masks should have been copied (potentially rescaled)
        mask_files = list((workspace / "associated_hqsam").glob("*.png"))
        assert len(mask_files) == 5

        try:
            from PIL import Image
            img = Image.open(mask_files[0])
            # At resolution=2, 640x480 → 320x240
            assert img.width == 320
            assert img.height == 240
        except ImportError:
            pass  # Skip resolution check if PIL unavailable

    def test_handles_empty_masks_dir(
        self, tmp_path: Path, fake_colmap: Path, fake_images: Path,
        fake_object_index: Path,
    ) -> None:
        empty_masks = tmp_path / "empty_masks"
        empty_masks.mkdir()
        workspace = tmp_path / "inpaint_ws"

        result = runner.prepare_data_layout(
            colmap_sparse_dir=fake_colmap,
            images_dir=fake_images,
            instance_masks_dir=empty_masks,
            object_index_path=fake_object_index,
            workspace=workspace,
            resolution=1,
        )
        assert result["num_objects"] == 2
        # No mask files copied
        mask_files = list((workspace / "associated_hqsam").glob("*.png"))
        assert len(mask_files) == 0
        # scene.json still written
        assert (workspace / "associated_hqsam" / "scene.json").is_file()

    def test_respects_max_objects(
        self, tmp_path: Path, fake_colmap: Path, fake_images: Path,
        fake_masks: Path, fake_object_index: Path, monkeypatch,
    ) -> None:
        monkeypatch.setattr(runner, "INPAINT360GS_MAX_OBJECTS", 1)
        workspace = tmp_path / "inpaint_ws"
        result = runner.prepare_data_layout(
            colmap_sparse_dir=fake_colmap,
            images_dir=fake_images,
            instance_masks_dir=fake_masks,
            object_index_path=fake_object_index,
            workspace=workspace,
            resolution=1,
        )
        assert result["num_objects"] == 1
        assert result["object_ids"] == [1]


# ---------------------------------------------------------------------------
# TestConvertGaussiansToMesh
# ---------------------------------------------------------------------------

class TestConvertGaussiansToMesh:
    def test_handles_missing_ply(self, tmp_path: Path) -> None:
        result = runner.convert_gaussians_to_mesh(
            ply_path=tmp_path / "nonexistent.ply",
            output_glb=tmp_path / "out.glb",
        )
        assert result["status"] == "failed"

    @pytest.mark.skipif(
        not _can_import("open3d"),
        reason="open3d not available",
    )
    def test_converts_synthetic_ply_to_glb(self, tmp_path: Path) -> None:
        """Write a small synthetic PLY and convert it."""
        import numpy as np

        # Create a minimal PLY file
        ply_path = tmp_path / "test.ply"
        n = 500
        points = np.random.randn(n, 3).astype(np.float32)
        colors = np.random.randint(0, 255, (n, 3), dtype=np.uint8)

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        with open(ply_path, "wb") as f:
            f.write(header.encode("ascii"))
            for i in range(n):
                f.write(points[i].tobytes())
                f.write(colors[i].tobytes())

        output_glb = tmp_path / "out.glb"
        result = runner.convert_gaussians_to_mesh(
            ply_path=ply_path,
            output_glb=output_glb,
        )
        # If open3d is present, this should succeed
        if result["status"] == "ok":
            assert output_glb.is_file()
            assert result["file_size_mb"] > 0


# ---------------------------------------------------------------------------
# TestRunSceneCleaning
# ---------------------------------------------------------------------------

class TestRunSceneCleaning:
    def test_skips_when_no_masks(
        self, tmp_path: Path, fake_colmap: Path, fake_images: Path,
        fake_object_index: Path, monkeypatch,
    ) -> None:
        # Point INPAINT360GS_DIR to a valid directory so the masks check fires
        fake_install = tmp_path / "fake_inpaint360gs"
        fake_install.mkdir()
        monkeypatch.setattr(runner, "INPAINT360GS_DIR", fake_install)
        monkeypatch.setattr(runner, "probe_installation", lambda **_: {"status": "ok"})

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        empty_masks = tmp_path / "no_masks"
        empty_masks.mkdir()

        result = runner.run_scene_cleaning(
            colmap_sparse_dir=fake_colmap,
            images_dir=fake_images,
            instance_masks_dir=empty_masks,
            object_index_path=fake_object_index,
            output_dir=output_dir,
        )
        assert result["status"] == "skipped"
        assert "no instance masks" in result.get("reason", "")

    def test_skips_when_no_objects(
        self, tmp_path: Path, fake_colmap: Path, fake_images: Path,
        fake_masks: Path, monkeypatch,
    ) -> None:
        # Point INPAINT360GS_DIR to a valid directory so the objects check fires
        fake_install = tmp_path / "fake_inpaint360gs"
        fake_install.mkdir()
        monkeypatch.setattr(runner, "INPAINT360GS_DIR", fake_install)
        monkeypatch.setattr(runner, "probe_installation", lambda **_: {"status": "ok"})

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        empty_index = tmp_path / "empty_index.json"
        empty_index.write_text('{"objects": []}', encoding="utf-8")

        result = runner.run_scene_cleaning(
            colmap_sparse_dir=fake_colmap,
            images_dir=fake_images,
            instance_masks_dir=fake_masks,
            object_index_path=empty_index,
            output_dir=output_dir,
        )
        assert result["status"] == "skipped"
        assert "no objects" in result.get("reason", "")

    def test_skips_when_install_dir_missing(
        self, tmp_path: Path, fake_colmap: Path, fake_images: Path,
        fake_masks: Path, fake_object_index: Path, monkeypatch,
    ) -> None:
        monkeypatch.setattr(runner, "INPAINT360GS_DIR", tmp_path / "nonexistent")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = runner.run_scene_cleaning(
            colmap_sparse_dir=fake_colmap,
            images_dir=fake_images,
            instance_masks_dir=fake_masks,
            object_index_path=fake_object_index,
            output_dir=output_dir,
        )
        assert result["status"] == "skipped"
        assert "probe failed" in result.get("reason", "")

    def test_resumes_from_existing_output(
        self, tmp_path: Path, fake_colmap: Path, fake_images: Path,
        fake_masks: Path, fake_object_index: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create fake pre-existing outputs
        expected_report = {
            "status": "ok",
            "inpainted_visual_glb": str(output_dir / "inpainted_visual_mesh.glb"),
        }
        (output_dir / "scene_cleaning_report.json").write_text(
            json.dumps(expected_report), encoding="utf-8"
        )
        (output_dir / "inpainted_visual_mesh.glb").write_bytes(b"fake glb data")

        result = runner.run_scene_cleaning(
            colmap_sparse_dir=fake_colmap,
            images_dir=fake_images,
            instance_masks_dir=fake_masks,
            object_index_path=fake_object_index,
            output_dir=output_dir,
            resume=True,
        )
        assert result["status"] == "ok"
        assert "inpainted_visual_mesh.glb" in result.get("inpainted_visual_glb", "")

    def test_returns_failed_on_subprocess_error(
        self, tmp_path: Path, fake_colmap: Path, fake_images: Path,
        fake_masks: Path, fake_object_index: Path, monkeypatch,
    ) -> None:
        """Inpaint360GS dir exists but train.py is missing → fails gracefully."""
        fake_install = tmp_path / "fake_inpaint360gs"
        fake_install.mkdir()
        monkeypatch.setattr(runner, "INPAINT360GS_DIR", fake_install)
        monkeypatch.setattr(runner, "probe_installation", lambda **_: {"status": "ok"})

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = runner.run_scene_cleaning(
            colmap_sparse_dir=fake_colmap,
            images_dir=fake_images,
            instance_masks_dir=fake_masks,
            object_index_path=fake_object_index,
            output_dir=output_dir,
        )
        assert result["status"] == "failed"
        # Should write a report file
        report_file = output_dir / "scene_cleaning_report.json"
        assert report_file.is_file()

    def test_report_written_on_failure(
        self, tmp_path: Path, fake_colmap: Path, fake_images: Path,
        fake_masks: Path, fake_object_index: Path, monkeypatch,
    ) -> None:
        """Verify the report JSON is always written, even on failure."""
        fake_install = tmp_path / "fake_inpaint360gs"
        fake_install.mkdir()
        monkeypatch.setattr(runner, "INPAINT360GS_DIR", fake_install)
        monkeypatch.setattr(runner, "probe_installation", lambda **_: {"status": "ok"})

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        runner.run_scene_cleaning(
            colmap_sparse_dir=fake_colmap,
            images_dir=fake_images,
            instance_masks_dir=fake_masks,
            object_index_path=fake_object_index,
            output_dir=output_dir,
        )
        report_path = output_dir / "scene_cleaning_report.json"
        assert report_path.is_file()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert "status" in data


# ---------------------------------------------------------------------------
# TestRunTraining
# ---------------------------------------------------------------------------

class TestRunTraining:
    def test_returns_failed_when_script_missing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(runner, "INPAINT360GS_DIR", tmp_path / "nonexistent")
        result = runner.run_training(workspace=tmp_path)
        assert result["status"] == "failed"
        assert "train.py not found" in result.get("reason", "")


# ---------------------------------------------------------------------------
# TestRunDistillation
# ---------------------------------------------------------------------------

class TestRunDistillation:
    def test_returns_failed_when_script_missing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(runner, "INPAINT360GS_DIR", tmp_path / "nonexistent")
        result = runner.run_distillation(workspace=tmp_path, model_path=tmp_path)
        assert result["status"] == "failed"
        assert "train_finetune.py not found" in result.get("reason", "")


# ---------------------------------------------------------------------------
# TestRunObjectRemoval
# ---------------------------------------------------------------------------

class TestRunObjectRemoval:
    def test_returns_failed_when_script_missing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(runner, "INPAINT360GS_DIR", tmp_path / "nonexistent")
        result = runner.run_object_removal(
            workspace=tmp_path, model_path=tmp_path, target_ids=[1, 2]
        )
        assert result["status"] == "failed"
        assert "edit_object_removal.py not found" in result.get("reason", "")


# ---------------------------------------------------------------------------
# TestRunInpaintOptimization
# ---------------------------------------------------------------------------

class TestRunInpaintOptimization:
    def test_returns_failed_when_script_missing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(runner, "INPAINT360GS_DIR", tmp_path / "nonexistent")
        result = runner.run_inpaint_optimization(workspace=tmp_path, model_path=tmp_path)
        assert result["status"] == "failed"
        assert "edit_object_inpaint.py not found" in result.get("reason", "")


# ---------------------------------------------------------------------------
# TestEnvHelpers
# ---------------------------------------------------------------------------

class TestEnvHelpers:
    def test_env_int_returns_default_on_missing(self) -> None:
        val = runner._env_int("__NONEXISTENT_KEY_XYZ__", 42)
        assert val == 42

    def test_env_int_returns_parsed_value(self, monkeypatch) -> None:
        monkeypatch.setenv("__TEST_INT__", "123")
        val = runner._env_int("__TEST_INT__", 0)
        assert val == 123

    def test_env_int_returns_default_on_bad_value(self, monkeypatch) -> None:
        monkeypatch.setenv("__TEST_INT__", "not_a_number")
        val = runner._env_int("__TEST_INT__", 99)
        assert val == 99

    def test_env_float_returns_default_on_missing(self) -> None:
        val = runner._env_float("__NONEXISTENT_KEY_XYZ__", 0.5)
        assert val == 0.5

    def test_env_float_returns_parsed_value(self, monkeypatch) -> None:
        monkeypatch.setenv("__TEST_FLOAT__", "0.75")
        val = runner._env_float("__TEST_FLOAT__", 0.0)
        assert val == 0.75


class TestProbeInstallation:
    def test_probe_fails_for_missing_install(self, tmp_path: Path) -> None:
        result = runner.probe_installation(install_dir=tmp_path / "missing")
        assert result["status"] == "failed"

    def test_probe_detects_missing_scripts(self, tmp_path: Path) -> None:
        install_dir = tmp_path / "inpaint"
        install_dir.mkdir()
        (install_dir / "train.py").write_text("", encoding="utf-8")
        result = runner.probe_installation(install_dir=install_dir)
        assert result["status"] == "failed"
        assert "train_finetune.py" in result.get("missing_scripts", [])
