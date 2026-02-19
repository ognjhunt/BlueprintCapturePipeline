"""Tests for occlusion-aware dimension completion in scripts/sam3_detect.py."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_sam3_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "sam3_detect.py"

    if "torch" not in sys.modules:
        sys.modules["torch"] = types.ModuleType("torch")

    spec = importlib.util.spec_from_file_location("sam3_detect_test_module", str(module_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_object(reference_crop: str) -> dict:
    return {
        "id": "box_1",
        "label": "box",
        "confidence": 0.8,
        "mean_confidence": 0.8,
        "n_frame_detections": 6,
        "n_total_detections": 8,
        "refinement": "da3_metric_depth",
        "boundingBox": {
            "center": [0.0, 0.0, 1.4],
            "extents": [0.45, 0.35, 0.3],
            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "orientationQuaternion": [1, 0, 0, 0],
        },
        "mean_box_px": [300.0, 220.0, 600.0, 640.0],
        "mean_centroid_px": [450.0, 430.0],
        "image_size": [960, 720],
        "reference_crop": reference_crop,
        "all_crops": [reference_crop],
    }


def test_dimension_completion_auto_skips_non_occluded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_sam3_module()
    crop_path = tmp_path / "object_crops" / "box.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake")

    obj = _base_object("object_crops/box.png")

    def _should_not_run(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("completion estimator should not run for non-occluded objects")

    monkeypatch.setattr(module, "_infer_dimension_completion_estimate", _should_not_run)

    updated, report = module._apply_occlusion_dimension_completion(
        objects=[obj],
        output_path=tmp_path / "object_point_cloud_index.json",
        environment="bedroom",
        mode_override="auto",
    )

    assert report["objects_attempted"] == 0
    assert updated[0]["dimension_completion"]["status"] == "skipped"
    assert updated[0]["dimension_completion"]["reason"] == "low_occlusion_score"


def test_dimension_completion_updates_occluded_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_sam3_module()
    crop_path = tmp_path / "object_crops" / "box.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake")

    obj = _base_object("object_crops/box.png")
    obj["mean_confidence"] = 0.55
    obj["n_frame_detections"] = 1
    obj["refinement"] = "heuristic_2d"
    obj["mean_box_px"] = [0.0, 180.0, 260.0, 690.0]  # touches left edge
    obj["boundingBox"]["extents"] = [0.4, 0.3, 0.25]

    monkeypatch.setattr(module, "_DIM_COMPLETION_MIN_OCCLUSION_SCORE", 0.2)
    monkeypatch.setattr(module, "_DIM_COMPLETION_MAX_EXPAND_RATIO", 1.8)

    def _fake_estimator(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "ok": True,
            "provider": "test_stub",
            "model": "test-model",
            "confidence": 0.92,
            "predicted_extents": [1.0, 0.9, 0.8],
            "reason": "synthetic completion",
        }

    monkeypatch.setattr(module, "_infer_dimension_completion_estimate", _fake_estimator)

    updated, report = module._apply_occlusion_dimension_completion(
        objects=[obj],
        output_path=tmp_path / "object_point_cloud_index.json",
        environment="bedroom",
        mode_override="auto",
    )

    assert report["objects_attempted"] == 1
    assert report["objects_completed"] == 1
    assert report["objects_updated"] == 1

    final_extents = updated[0]["boundingBox"]["extents"]
    assert final_extents[0] > 0.4
    assert final_extents[1] > 0.3
    assert final_extents[2] > 0.25
    assert final_extents[0] <= (0.4 * 1.8) + 1e-4
    assert updated[0]["dimension_completion"]["status"] == "completed"
    assert updated[0]["dimension_completion"]["provider"] == "test_stub"


def test_dimension_completion_skips_low_model_confidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_sam3_module()
    crop_path = tmp_path / "object_crops" / "box.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake")

    obj = _base_object("object_crops/box.png")
    obj["mean_confidence"] = 0.5
    obj["n_frame_detections"] = 1
    obj["refinement"] = "heuristic_2d"
    obj["mean_box_px"] = [0.0, 120.0, 220.0, 700.0]
    observed = list(obj["boundingBox"]["extents"])

    monkeypatch.setattr(module, "_DIM_COMPLETION_MIN_OCCLUSION_SCORE", 0.2)
    monkeypatch.setattr(module, "_DIM_COMPLETION_MIN_CONFIDENCE", 0.35)

    def _low_conf_estimator(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "ok": True,
            "provider": "test_stub",
            "model": "test-model",
            "confidence": 0.12,
            "predicted_extents": [0.9, 0.8, 0.7],
        }

    monkeypatch.setattr(module, "_infer_dimension_completion_estimate", _low_conf_estimator)

    updated, report = module._apply_occlusion_dimension_completion(
        objects=[obj],
        output_path=tmp_path / "object_point_cloud_index.json",
        environment="bedroom",
        mode_override="auto",
    )

    assert report["objects_attempted"] == 1
    assert report["objects_completed"] == 0
    assert updated[0]["dimension_completion"]["status"] == "skipped"
    assert updated[0]["dimension_completion"]["reason"] == "model_confidence_below_threshold"
    assert updated[0]["boundingBox"]["extents"] == observed
