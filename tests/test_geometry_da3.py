from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline import geometry_da3 as da3


def test_da3_probe_timestamp_extract_and_frame_helpers(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    assert da3._safe_float("bad", 1.5) == 1.5
    assert da3._frame_count_from_probe({"duration_seconds": 0, "avg_frame_rate": ""}) == 6
    assert da3._frame_count_from_probe({"duration_seconds": 3, "avg_frame_rate": "30/1"}) == 6
    assert da3._frame_count_from_probe({"duration_seconds": 20, "avg_frame_rate": "bad"}) == 20
    assert da3._sample_timestamps({"duration_seconds": 0}, "streaming") == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
    ]
    assert da3._sample_timestamps({"duration_seconds": 1, "avg_frame_rate": "1"}, "batch") == [
        0.0,
        0.5,
        1.0,
    ]
    original_frame_count = da3._frame_count_from_probe
    monkeypatch.setattr(da3, "_frame_count_from_probe", lambda _probe: 1)
    assert da3._sample_timestamps({"duration_seconds": 1}, "batch") == [0.0]
    monkeypatch.setattr(da3, "_frame_count_from_probe", original_frame_count)

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    extracted = tmp_path / "frame.png"

    def fake_run_success(command, **_kwargs):  # type: ignore[no-untyped-def]
        Path(command[-1]).write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(da3.subprocess, "run", fake_run_success)
    assert da3._extract_frame(video, 1.25, extracted) is True
    monkeypatch.setattr(da3.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1))
    assert da3._extract_frame(video, 0.0, tmp_path / "failed.png") is False
    monkeypatch.setattr(
        da3.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("ffmpeg")),
    )
    assert da3._extract_frame(video, 0.0, tmp_path / "missing.png") is False

    gradient = tmp_path / "gradient.png"
    da3._write_gradient_frame(gradient, width=8, height=8, frame_index=2)
    assert gradient.is_file()
    monkeypatch.setattr(da3, "Image", None)
    npy_path = tmp_path / "gradient.npy"
    da3._write_gradient_frame(npy_path, width=8, height=8, frame_index=3)
    assert np.load(npy_path).shape == (64, 64, 3)


def test_da3_sample_frames_runtime_loading_and_inference(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(da3, "Image", Image)
    def fake_extract(_video, _timestamp, path):  # type: ignore[no-untyped-def]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return True

    monkeypatch.setattr(da3, "_extract_frame", fake_extract)
    frames, warnings = da3._sample_frames(
        video_path=tmp_path / "video.mp4",
        frames_dir=tmp_path / "frames",
        video_probe={"duration_seconds": 0.5, "avg_frame_rate": "2", "width": 80, "height": 60},
        execution_mode="batch",
    )
    assert warnings == []
    assert frames[0]["is_keyframe"] is True
    monkeypatch.setattr(da3, "_extract_frame", lambda *_args, **_kwargs: False)
    frames, warnings = da3._sample_frames(
        video_path=tmp_path / "video.mp4",
        frames_dir=tmp_path / "fallback",
        video_probe={"duration_seconds": 0, "width": 32, "height": 32},
        execution_mode="batch",
    )
    assert "video_decode_unavailable:synthetic_frames_used" in warnings
    assert Path(frames[0]["image_path"]).is_file()

    runtime, runtime_warnings = da3._load_da3_runtime("metric")
    assert runtime is None
    assert runtime_warnings[0].startswith(
        ("da3_runtime_unavailable:", "da3_model_load_failed:")
    )

    fake_module = types.ModuleType("depth_anything_3.api")

    class FakeDepthAnything3:
        @staticmethod
        def from_pretrained(_path: str, *, model_name: str):  # type: ignore[no-untyped-def]
            return {"model": model_name}

    fake_module.DepthAnything3 = FakeDepthAnything3
    package = types.ModuleType("depth_anything_3")
    package.api = fake_module
    monkeypatch.setitem(sys.modules, "depth_anything_3", package)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", fake_module)
    runtime, runtime_warnings = da3._load_da3_runtime("configured")
    assert runtime == {"model": "da3metric-large"}
    assert runtime_warnings == ["da3_model_loaded:configured"]

    class FailingDepthAnything3:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("load failed")

    fake_module.DepthAnything3 = FailingDepthAnything3
    runtime, runtime_warnings = da3._load_da3_runtime("configured")
    assert runtime is None
    assert runtime_warnings == ["da3_model_load_failed:RuntimeError"]

    rgb = np.ones((2, 2, 3), dtype=np.float32)

    class RuntimeMapping:
        def infer_image(self, _rgb):  # type: ignore[no-untyped-def]
            return {"depth": np.ones((1, 2, 2), dtype=np.float32)}

    assert da3._infer_depth_with_runtime(RuntimeMapping(), rgb).shape == (2, 2)

    class RuntimeFallback:
        def infer_image(self, _rgb):  # type: ignore[no-untyped-def]
            raise RuntimeError("bad")

        def predict(self, _rgb):  # type: ignore[no-untyped-def]
            return None

        def __call__(self, _rgb):  # type: ignore[no-untyped-def]
            return np.ones((2, 2), dtype=np.float32)

    assert da3._infer_depth_with_runtime(RuntimeFallback(), rgb).shape == (2, 2)
    assert da3._infer_depth_with_runtime(types.SimpleNamespace(predict=lambda _rgb: np.ones((2, 2, 2, 2))), rgb) is None


def test_da3_depth_artifacts_frame_loading_and_provider(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rgb = np.asarray([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    png = tmp_path / "frame.png"
    Image.fromarray(rgb, mode="RGB").save(png)
    loaded, width, height = da3._load_frame_rgb(png)
    assert (width, height) == (2, 1)
    assert loaded.shape == (1, 2, 3)
    npy_2d = tmp_path / "frame.npy"
    np.save(npy_2d, np.ones((2, 2), dtype=np.float32))
    loaded, width, height = da3._load_frame_rgb(npy_2d)
    assert loaded.shape == (2, 2, 3)
    assert (width, height) == (2, 2)
    monkeypatch.setattr(da3, "Image", None)
    loaded, width, height = da3._load_frame_rgb(tmp_path / "anything.bin")
    assert loaded.shape == (64, 64, 3)
    assert (width, height) == (64, 64)
    monkeypatch.setattr(da3, "Image", Image)

    frame_records = [{"frame_index": 0, "image_path": str(png)}]
    monkeypatch.setattr(da3, "_load_da3_runtime", lambda _model: (None, ["runtime-missing"]))
    updated, warnings, metrics = da3._write_depth_confidence_artifacts(
        frame_records=frame_records,
        depth_dir=tmp_path / "depth",
        confidence_dir=tmp_path / "confidence",
        model="metric",
    )
    assert warnings == ["runtime-missing"]
    assert metrics["backend"] == "synthetic_fallback"
    assert Path(updated[0]["depth_path"]).is_file()
    assert updated[0]["metric_depth_truth"] is False
    assert updated[0]["depth_measurement_source"] == "monocular_depth_estimate"

    class RuntimeNoDepth:
        def infer_image(self, _rgb):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(da3, "_load_da3_runtime", lambda _model: (RuntimeNoDepth(), []))
    updated, warnings, metrics = da3._write_depth_confidence_artifacts(
        frame_records=[{"frame_index": 1, "image_path": str(png)}],
        depth_dir=tmp_path / "depth2",
        confidence_dir=tmp_path / "confidence2",
        model="metric",
    )
    assert metrics["backend"] == "da3_python_runtime"
    assert warnings == ["da3_frame_inference_failed:frame_000001:synthetic_depth_used"]
    assert updated[0]["metric_depth_truth"] is False

    assert da3._intrinsics_from_probe({"width": 100, "height": 50})["fx"] == 92.0
    pose = da3._pose_for_frame(2, 1.0)
    assert pose[0][0][3] == 0.36

    monkeypatch.setattr(
        da3,
        "_sample_frames",
        lambda **_kwargs: ([{"frame_index": 0, "timestamp_seconds": 0.0, "image_path": str(png), "is_keyframe": True}], []),
    )
    monkeypatch.setattr(da3, "_load_da3_runtime", lambda _model: (None, []))
    result = da3.run_da3_provider(
        video_path=tmp_path / "video.mp4",
        geometry_root=tmp_path / "geom",
        video_probe={"width": 100, "height": 50},
        provider="da3",
        model="metric",
        execution_mode="batch",
    )
    assert result["provider"] == "da3"
    assert result["keyframe_indices"] == [0]
    assert result["frames"][0]["pose_confidence"] == 0.0
    assert result["frames"][0]["metric_depth_truth"] is False
    assert result["frames"][0]["metric_pose_truth"] is False
    assert result["frames"][0]["pose_measurement_source"] == (
        "synthetic_trajectory_placeholder"
    )
    assert result["intrinsics"]["metric_intrinsics_truth"] is False
    assert result["intrinsics"]["intrinsics_measurement_source"] == (
        "heuristic_from_image_dimensions"
    )
    assert result["qualification_role"] == "diagnostic_cross_check_only"
    assert result["metric_geometry_authority"] is False
