"""Geometry-aware camera framing expectations.

Pins the contract that repaired the scene 839873 construction refusal-in-
waiting: the sealed external camera (1.43 m from a 12.8 cm object, fx=172.9,
320x180) can project at most ~245 px^2 of bounding box, while the configured
framing gate demanded 200 segmented pixels -- a threshold that geometry could
never meet and that would have failed every paid construction run of that
scene after full execution.  The constants here are copied from the sealed
production scene plan, so a regression in the projection or the scaling shows
up against the exact artifact that motivated it.
"""

from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.native_task_camera_framing_expectation import (
    FRAMING_MINIMUM_EXPECTED_BBOX_AREA_PX,
    FRAMING_MINIMUM_FLOOR_PIXELS,
    NativeTaskCameraFramingExpectationError,
    camera_framing_expectation,
    effective_framing_minimums,
    measure_task_object_extent_m,
)
from blueprint_pipeline.native_task_camera_observability import (
    measure_native_task_camera_observability,
)


# Copied verbatim from the sealed scene 839873 production scene plan
# (native_task_arena_scene_plan.v1, plan for run r4 of the final batch).
SCENE_839873_EXTERNAL_CAMERA = {
    "role": "external",
    "policy_input": True,
    "pose_frame": "world",
    "frame_from_camera_matrix": [
        -0.9998820883203204,
        0.01379617788430592,
        -0.0067435103602981,
        2.9259961206833616,
        0.015356088570819706,
        0.8983115127384338,
        -0.43909067016436565,
        -6.132663812912105,
        0.0,
        -0.439142450188286,
        -0.8984174466486214,
        2.102958,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    "intrinsics": {
        "fx": 172.88839142740494,
        "fy": 172.88839142740494,
        "cx": 159.5,
        "cy": 89.5,
        "width": 320,
        "height": 180,
    },
}
# Measured from the staged replacement asset (12.7 x 13.1 x 12.8 cm mug).
SCENE_839873_OBJECT_EXTENT_M = [0.127, 0.1311, 0.128]
SCENE_839873_START_POSITION = [2.9742285, -6.7605156, 0.818319]
SCENE_839873_TARGET_POSITION = [3.0942285, -6.7605156, 0.818319]


def _cube_usda(tmp_path, *, size: float = 0.12) -> str:
    path = tmp_path / "object.usda"
    half = size / 2.0
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset"
{{
    def Mesh "body"
    {{
        point3f[] points = [(-{half}, -{half}, 0.0), ({half}, {half}, {size})]
    }}
}}
''',
        encoding="utf-8",
    )
    return str(path)


def test_extent_measurement_reads_authored_geometry(tmp_path) -> None:
    extent = measure_task_object_extent_m(_cube_usda(tmp_path, size=0.12))
    assert extent == pytest.approx([0.12, 0.12, 0.12])


def test_extent_measurement_refuses_geometry_free_assets(tmp_path) -> None:
    path = tmp_path / "empty.usda"
    path.write_text("#usda 1.0\n", encoding="utf-8")
    with pytest.raises(
        NativeTaskCameraFramingExpectationError,
        match="native_task_camera_framing_task_object_extent_unmeasurable",
    ):
        measure_task_object_extent_m(path)


def test_scene_839873_external_expectation_matches_measured_run() -> None:
    """The projection must reproduce what construction run r18 measured.

    r18 segmented 93 task-object pixels on this camera at this geometry.  The
    projected bounding-box area must land near 240 px^2 (the raked-view mask
    fills ~39% of it), and the effective minimum must scale below 93 while the
    configured 200 stays out of reach of the geometry.
    """

    expectation = camera_framing_expectation(
        camera=SCENE_839873_EXTERNAL_CAMERA,
        object_extent_m=SCENE_839873_OBJECT_EXTENT_M,
        object_positions_world=[
            SCENE_839873_START_POSITION,
            SCENE_839873_TARGET_POSITION,
        ],
    )
    assert expectation is not None
    assert expectation["expected_bbox_area_px"] == pytest.approx(245.5, abs=2.0)
    for row in expectation["positions"]:
        assert row["depth_m"] == pytest.approx(1.43, abs=0.01)
    effective = effective_framing_minimums(
        minimum_pixels=200,
        minimum_pixel_fraction=0.003,
        frame_width=320,
        frame_height=180,
        expected_bbox_area_px=expectation["expected_bbox_area_px"],
    )
    assert effective["effective_minimum_pixels"] == 50
    assert 93 >= effective["effective_minimum_pixels"]
    assert 200 > expectation["expected_bbox_area_px"] * 0.39


def test_scaling_only_lowers_and_respects_the_floor() -> None:
    # Geometry that supports the configured constants keeps them bit-exactly.
    large = effective_framing_minimums(
        minimum_pixels=200,
        minimum_pixel_fraction=0.003,
        frame_width=320,
        frame_height=180,
        expected_bbox_area_px=5000.0,
    )
    assert large["effective_minimum_pixels"] == 200
    assert large["effective_minimum_pixel_fraction"] == 0.003
    # A tiny projection clamps at the floor rather than reaching zero.
    tiny = effective_framing_minimums(
        minimum_pixels=200,
        minimum_pixel_fraction=0.003,
        frame_width=320,
        frame_height=180,
        expected_bbox_area_px=FRAMING_MINIMUM_EXPECTED_BBOX_AREA_PX,
    )
    assert tiny["effective_minimum_pixels"] == FRAMING_MINIMUM_FLOOR_PIXELS


def test_robot_parented_cameras_carry_no_expectation() -> None:
    wrist = dict(SCENE_839873_EXTERNAL_CAMERA, role="wrist", pose_frame="robot_body")
    assert (
        camera_framing_expectation(
            camera=wrist,
            object_extent_m=SCENE_839873_OBJECT_EXTENT_M,
            object_positions_world=[SCENE_839873_START_POSITION],
        )
        is None
    )


def test_object_behind_camera_is_refused() -> None:
    behind = [
        SCENE_839873_EXTERNAL_CAMERA["frame_from_camera_matrix"][3],
        SCENE_839873_EXTERNAL_CAMERA["frame_from_camera_matrix"][7],
        SCENE_839873_EXTERNAL_CAMERA["frame_from_camera_matrix"][11] + 1.0,
    ]
    with pytest.raises(
        NativeTaskCameraFramingExpectationError,
        match="native_task_camera_framing_object_behind_camera",
    ):
        camera_framing_expectation(
            camera=SCENE_839873_EXTERNAL_CAMERA,
            object_extent_m=SCENE_839873_OBJECT_EXTENT_M,
            object_positions_world=[behind],
        )


def _frame_and_mask(pixel_count: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """A textured frame whose centred target mask has ``pixel_count`` pixels."""

    rng = np.random.default_rng(11)
    rgb = rng.integers(20, 235, size=(180, 320, 3), dtype=np.uint8)
    semantic = np.zeros((180, 320), dtype=np.int64)
    side = int(np.ceil(np.sqrt(pixel_count)))
    y0, x0 = 90 - side // 2, 160 - side // 2
    coords = [
        (y0 + index // side, x0 + index % side) for index in range(pixel_count)
    ]
    for y, x in coords:
        semantic[y, x] = 7
    labels = {"7": {"class": "task_object"}}
    return semantic, rgb, labels


def test_gate_accepts_the_measured_r18_frame_only_with_the_expectation() -> None:
    """93 centred pixels: refused under the constant, accepted under geometry."""

    semantic, rgb, labels = _frame_and_mask(93)
    without = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels=labels,
        rgb=rgb,
        site_appearance_render_expected=False,
        minimum_pixels=200,
        minimum_pixel_fraction=0.003,
    )
    assert without["passed"] is False
    assert "native_task_camera_semantic_framing_below_threshold" in without["blockers"]
    expectation = camera_framing_expectation(
        camera=SCENE_839873_EXTERNAL_CAMERA,
        object_extent_m=SCENE_839873_OBJECT_EXTENT_M,
        object_positions_world=[
            SCENE_839873_START_POSITION,
            SCENE_839873_TARGET_POSITION,
        ],
    )
    with_geometry = measure_native_task_camera_observability(
        semantic_ids=semantic,
        id_to_labels=labels,
        rgb=rgb,
        site_appearance_render_expected=False,
        minimum_pixels=200,
        minimum_pixel_fraction=0.003,
        framing_expectation=expectation,
    )
    assert with_geometry["passed"] is True
    assert with_geometry["thresholds"]["effective_minimum_pixels"] == 50
    assert with_geometry["framing_thresholds"]["configured_minimum_pixels"] == 200
    # An object occluded down to speckle still fails under the scaled gate.
    sparse_semantic, sparse_rgb, sparse_labels = _frame_and_mask(20)
    occluded = measure_native_task_camera_observability(
        semantic_ids=sparse_semantic,
        id_to_labels=sparse_labels,
        rgb=sparse_rgb,
        site_appearance_render_expected=False,
        minimum_pixels=200,
        minimum_pixel_fraction=0.003,
        framing_expectation=expectation,
    )
    assert occluded["passed"] is False
