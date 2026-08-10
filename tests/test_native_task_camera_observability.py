from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.native_task_camera_observability import (
    NativeTaskCameraObservabilityError,
    configure_native_semantic_id_output,
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
    assert result["semantic_input"] == {
        "input_shape": [100, 200],
        "input_dtype": "int32",
        "representation": "integer_id_map_2d",
    }


def test_single_channel_semantic_ids_cover_current_arena_output_contract() -> None:
    semantic = np.zeros((40, 60, 1), dtype=np.int32)
    semantic[10:30, 20:40, 0] = 11

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={"11": {"class": "task_object"}},
        minimum_pixels=100,
        minimum_pixel_fraction=0.01,
    )

    assert result["pixel_count"] == 400
    assert result["semantic_input"]["representation"] == (
        "integer_id_map_single_channel"
    )
    assert result["passed"] is True


def test_single_channel_ids_resolve_replicator_rgba_tuple_label_keys() -> None:
    """The pinned renderer can pair int32 pixels with packed-RGBA info keys."""

    rgba = (240, 4, 111, 255)
    packed_little = rgba[0] | (rgba[1] << 8) | (rgba[2] << 16) | (rgba[3] << 24)
    packed_signed = packed_little - (1 << 32)
    semantic = np.zeros((40, 60, 1), dtype=np.int32)
    semantic[10:30, 20:40, 0] = packed_signed

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={str(rgba): {"class": "task_object"}},
        minimum_pixels=100,
        minimum_pixel_fraction=0.01,
    )

    assert result["target_semantic_ids"] == [packed_signed]
    assert result["pixel_count"] == 400
    assert result["semantic_identifier_resolution"] == {
        "scalar_target_identifier_count": 0,
        "rgba_tuple_target_identifier_count": 1,
        "rgba_tuple_encoding": "rgba_little_endian_int32",
        "rgba_tuple_encoding_evidence_match_count": 1,
        "resolution_authority": "native_integer_aov_values_and_id_to_labels",
    }
    assert result["passed"] is True


def test_rgba_tuple_encoding_is_resolved_from_other_visible_labels() -> None:
    """An off-camera target remains a measured zero instead of an endian guess."""

    target = (240, 4, 111, 255)
    scene = (1, 2, 3, 4)
    scene_packed = scene[0] | (scene[1] << 8) | (scene[2] << 16) | (scene[3] << 24)
    semantic = np.full((20, 20), scene_packed, dtype=np.uint32)

    result = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels={
            str(target): {"class": "task_object"},
            str(scene): {"class": "scene_collision"},
        },
        minimum_pixels=1,
        minimum_pixel_fraction=0.001,
    )

    target_packed = target[0] | (target[1] << 8) | (target[2] << 16) | (target[3] << 24)
    assert result["target_semantic_ids"] == [target_packed]
    assert result["pixel_count"] == 0
    assert result["passed"] is False


def test_rgba_tuple_encoding_without_native_pixel_evidence_fails_closed() -> None:
    semantic = np.zeros((20, 20), dtype=np.int32)

    with pytest.raises(NativeTaskCameraObservabilityError) as excinfo:
        measure_native_task_camera_observability(
            semantic_ids=semantic,
            id_to_labels={"(240, 4, 111, 255)": {"class": "task_object"}},
            minimum_pixels=1,
            minimum_pixel_fraction=0.001,
        )

    assert excinfo.value.errors == (
        "native_task_camera_semantic_tuple_encoding_unresolved",
    )


def test_colorized_arena_semantics_cannot_be_misread_as_class_ids() -> None:
    colorized = np.zeros((40, 60, 4), dtype=np.uint8)

    with pytest.raises(NativeTaskCameraObservabilityError) as excinfo:
        measure_native_task_camera_observability(
            semantic_ids=colorized,
            id_to_labels={"11": {"class": "task_object"}},
            minimum_pixels=100,
            minimum_pixel_fraction=0.01,
        )

    assert excinfo.value.errors == (
        "native_task_camera_semantic_output_colorized",
    )


def test_semantic_id_output_configures_legacy_and_renderer_owned_controls() -> None:
    camera_cfg = SimpleNamespace(
        colorize_semantic_segmentation=True,
        renderer_cfg=SimpleNamespace(colorize_semantic_segmentation=True),
    )

    receipt = configure_native_semantic_id_output(camera_cfg)

    assert camera_cfg.colorize_semantic_segmentation is False
    assert camera_cfg.renderer_cfg.colorize_semantic_segmentation is False
    assert receipt["configured_controls"] == [
        "camera_cfg.colorize_semantic_segmentation",
        "camera_cfg.renderer_cfg.colorize_semantic_segmentation",
    ]
    assert receipt["passed"] is True


def test_semantic_id_output_supports_original_legacy_camera_fixture() -> None:
    camera_cfg = SimpleNamespace(colorize_semantic_segmentation=True)

    receipt = configure_native_semantic_id_output(camera_cfg)

    assert camera_cfg.colorize_semantic_segmentation is False
    assert receipt["configured_controls"] == [
        "camera_cfg.colorize_semantic_segmentation"
    ]


def test_semantic_id_output_supports_renderer_owned_only_camera_config() -> None:
    camera_cfg = SimpleNamespace(
        renderer_cfg=SimpleNamespace(colorize_semantic_segmentation=True)
    )

    receipt = configure_native_semantic_id_output(camera_cfg)

    assert camera_cfg.renderer_cfg.colorize_semantic_segmentation is False
    assert receipt["configured_controls"] == [
        "camera_cfg.renderer_cfg.colorize_semantic_segmentation"
    ]


def test_semantic_id_output_fails_when_runtime_exposes_no_representation_control() -> None:
    with pytest.raises(NativeTaskCameraObservabilityError) as excinfo:
        configure_native_semantic_id_output(SimpleNamespace())

    assert excinfo.value.errors == (
        "native_task_camera_semantic_id_configuration_control_missing",
    )


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
