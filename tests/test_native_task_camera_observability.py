from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.native_task_camera_observability import (
    NativeTaskCameraObservabilityError,
    measure_native_task_camera_entity_observability,
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
    assert result["bbox_within_margin"] is True
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


def test_bbox_edge_clearance_rejects_a_clipped_entity_with_centered_centroid() -> None:
    semantic = np.full((100, 100), 4, dtype=np.int32)

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"4": {"class": "task_object"}},
        minimum_pixels=100,
        minimum_pixel_fraction=0.01,
        centroid_margin_fraction=0.05,
    )

    assert result["centroid_within_margin"] is True
    assert result["bbox_within_margin"] is False
    assert result["passed"] is False


def test_multi_entity_gate_requires_movable_and_destination_by_identity() -> None:
    semantic = np.zeros((100, 160), dtype=np.int32)
    semantic[25:65, 25:65] = 7
    semantic[30:80, 95:145] = 9

    result = measure_native_task_camera_entity_observability(
        semantic_ids=semantic,
        id_to_labels={
            "7": {"class": "movable_deformable", "entity_id": "cloth"},
            "9": {"class": "destination_receptacle", "entity_id": "basket"},
            "11": {"class": "obstacle", "entity_id": "obstacle_a"},
            "12": {"class": "obstacle", "entity_id": "obstacle_b"},
        },
        entity_requirements=[
            {
                "entity_id": "cloth",
                "minimum_pixels": 500,
                "minimum_pixel_fraction": 0.02,
            },
            {
                "entity_id": "basket",
                "minimum_pixels": 500,
                "minimum_pixel_fraction": 0.02,
            },
        ],
    )

    assert result["required_entity_ids"] == ["basket", "cloth"]
    assert result["all_entities_passed"] is True
    assert result["passed"] is True
    assert result["pixel_count"] == 4100
    by_id = {row["entity_id"]: row for row in result["entity_observability"]}
    assert by_id["cloth"]["target_semantic_ids"] == [7]
    assert by_id["basket"]["target_semantic_ids"] == [9]
    assert result["semantic_roles_used_as_entity_identity"] is False


def test_class_pixels_cannot_substitute_for_missing_entity_id() -> None:
    semantic = np.full((40, 40), 3, dtype=np.int32)

    result = measure_native_task_camera_entity_observability(
        semantic_ids=semantic,
        id_to_labels={"3": {"class": "destination_receptacle", "entity_id": "other_basket"}},
        entity_requirements=[
            {
                "entity_id": "frozen_basket",
                "minimum_pixels": 1,
                "minimum_pixel_fraction": 0.001,
            }
        ],
    )

    assert result["entity_observability"][0]["target_semantic_ids"] == []
    assert result["entity_observability"][0]["pixel_count"] == 0
    assert result["passed"] is False


def test_semantic_identifier_aliases_cannot_satisfy_two_entities() -> None:
    semantic = np.full((40, 40), 7, dtype=np.int32)

    with pytest.raises(
        NativeTaskCameraObservabilityError,
        match="native_task_camera_semantic_identifier_alias",
    ):
        measure_native_task_camera_entity_observability(
            semantic_ids=semantic,
            id_to_labels={
                7: {"entity_id": "cloth"},
                "7": {"entity_id": "basket"},
            },
            entity_requirements=[
                {
                    "entity_id": "cloth",
                    "minimum_pixels": 1,
                    "minimum_pixel_fraction": 0.001,
                },
                {
                    "entity_id": "basket",
                    "minimum_pixels": 1,
                    "minimum_pixel_fraction": 0.001,
                },
            ],
        )


@pytest.mark.parametrize(
    "semantic",
    [
        np.full((4, 4), 7.9, dtype=np.float32),
        np.full((4, 4), True, dtype=np.bool_),
    ],
)
def test_non_integer_semantic_planes_fail_closed(semantic: np.ndarray) -> None:
    with pytest.raises(
        NativeTaskCameraObservabilityError,
        match="native_task_camera_semantic_shape_invalid",
    ):
        measure_native_task_camera_observability(
            semantic_ids=semantic,
            id_to_labels={"7": {"class": "task_object"}},
            minimum_pixels=1,
            minimum_pixel_fraction=0.001,
        )


@pytest.mark.parametrize(
    "requirements",
    [
        [],
        [{"entity_id": 7, "minimum_pixels": 1, "minimum_pixel_fraction": 0.1}],
        [
            {"entity_id": "same", "minimum_pixels": 1, "minimum_pixel_fraction": 0.1},
            {"entity_id": "same", "minimum_pixels": 1, "minimum_pixel_fraction": 0.1},
        ],
        [{"entity_id": "cloth", "minimum_pixels": True, "minimum_pixel_fraction": 0.1}],
        [{"entity_id": "cloth", "minimum_pixels": 1, "minimum_pixel_fraction": 1.1}],
    ],
)
def test_entity_requirement_boundary_is_strict(requirements: list[dict]) -> None:
    with pytest.raises(NativeTaskCameraObservabilityError):
        measure_native_task_camera_entity_observability(
            semantic_ids=np.zeros((4, 4), dtype=np.int32),
            id_to_labels={},
            entity_requirements=requirements,
        )
