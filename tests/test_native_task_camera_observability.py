from __future__ import annotations

import numpy as np

from blueprint_pipeline.native_task_camera_observability import (
    measure_native_task_camera_observability,
)


def test_exact_semantic_pixels_gate_visibility_and_framing() -> None:
    semantic = np.zeros((100, 200), dtype=np.int32)
    semantic[30:70, 70:130] = 7

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"7": {"class": "task_object"}},
        minimum_pixels=200,
        minimum_pixel_fraction=0.005,
    )

    assert result["pixel_count"] == 2400
    assert result["bbox_xyxy"] == [70, 30, 129, 69]
    assert result["centroid_within_margin"] is True
    assert result["passed"] is True


def test_wrong_scene_label_cannot_be_counted_as_the_task() -> None:
    semantic = np.full((20, 20), 3, dtype=np.int32)

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"3": {"class": "approved_can"}},
        minimum_pixels=1,
        minimum_pixel_fraction=0.001,
    )

    assert result["target_semantic_ids"] == []
    assert result["pixel_count"] == 0
    assert result["passed"] is False


def test_a_large_but_off_frame_centroid_fails_framing() -> None:
    semantic = np.zeros((100, 100), dtype=np.int32)
    semantic[10:90, :20] = 4

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"4": {"class": "task_object"}},
        minimum_pixels=100,
        minimum_pixel_fraction=0.01,
        centroid_margin_fraction=0.15,
    )

    assert result["pixel_count"] == 1600
    assert result["centroid_within_margin"] is False
    assert result["passed"] is False
