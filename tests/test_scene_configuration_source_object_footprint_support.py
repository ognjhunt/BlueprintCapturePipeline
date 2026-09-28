"""The admitted removal support must cover the object's full visual footprint.

Scene 839873, 2026-08-29: the independent reviewer rejected six of eight frames
for a "distinct residual oval/shadow" on the tabletop.  Measured against the
run's own artifacts, the raw ``gpt-image-2`` edit was clean and the oval was
reintroduced by our own locality seal: the admitted support is the object's
tight projected AABB, and ``_inner_feather_alpha`` erodes
``MAX_INNER_FEATHER_RADIUS_PIXELS`` inward from its boundary, restoring source
pixels in that band.  A contact shadow sits at the object's base, inside the
band, so it was deterministically put back before ArtiFixer ever trained.

The support therefore has to admit the object plus a bounded contact-shadow
margin, and that margin has to clear the feather band, or the feather simply
eats it again.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
    MAX_SOURCE_OBJECT_FOOTPRINT_MARGIN_PIXELS,
    MIN_SOURCE_OBJECT_FOOTPRINT_MARGIN_PIXELS,
    _project_registered_bounds_mask,
)
from blueprint_pipeline.task_evaluation_scene_configuration_semantic_locality import (
    MAX_INNER_FEATHER_RADIUS_PIXELS,
)


def _camera(width: int = 256, height: int = 256) -> dict:
    return {
        "T_world_camera_provider_frame": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": 200.0,
            "fy": 200.0,
            "cx": width / 2,
            "cy": height / 2,
            "near": 0.01,
        },
    }


def _frame(tmp_path, width: int = 256, height: int = 256):
    path = tmp_path / "frame.png"
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(path)
    return path


def _mask_bbox(path):
    array = np.asarray(Image.open(path).convert("L"))
    ys, xs = np.nonzero(array > 127)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def test_margin_clears_the_inner_feather_band() -> None:
    """A margin inside the feather radius would be eroded straight back."""

    assert (
        MIN_SOURCE_OBJECT_FOOTPRINT_MARGIN_PIXELS > MAX_INNER_FEATHER_RADIUS_PIXELS
    )
    assert (
        MAX_SOURCE_OBJECT_FOOTPRINT_MARGIN_PIXELS
        >= MIN_SOURCE_OBJECT_FOOTPRINT_MARGIN_PIXELS
    )


def test_support_admits_a_bounded_contact_shadow_margin(tmp_path) -> None:
    frame_path = _frame(tmp_path)
    receipt = _project_registered_bounds_mask(
        minimum_xyz=[-0.05, -0.05, 1.0],
        maximum_xyz=[0.05, 0.05, 1.2],
        camera=_camera(),
        frame_path=frame_path,
        output_path=tmp_path / "mask.png",
    )

    margin = receipt["contact_shadow_margin_pixels"]
    assert MIN_SOURCE_OBJECT_FOOTPRINT_MARGIN_PIXELS <= margin
    assert margin <= MAX_SOURCE_OBJECT_FOOTPRINT_MARGIN_PIXELS
    # The margin must clear the band the seal erodes, or it is restored anyway.
    assert margin > MAX_INNER_FEATHER_RADIUS_PIXELS

    tight = receipt["object_pixel_bounds_xyxy"]
    admitted = receipt["pixel_bounds_xyxy"]
    assert admitted[0] == tight[0] - margin
    assert admitted[1] == tight[1] - margin
    assert admitted[2] == tight[2] + margin
    assert admitted[3] == tight[3] + margin
    # The written mask is the admitted support, not the tight object box.
    assert _mask_bbox(tmp_path / "mask.png") == tuple(admitted)
    assert receipt["foreground_pixel_count"] == (admitted[2] - admitted[0]) * (
        admitted[3] - admitted[1]
    )


def test_object_stays_strictly_inside_the_admitted_support(tmp_path) -> None:
    """The feather may only ever eat margin, never the object itself."""

    receipt = _project_registered_bounds_mask(
        minimum_xyz=[-0.05, -0.05, 1.0],
        maximum_xyz=[0.05, 0.05, 1.2],
        camera=_camera(),
        frame_path=_frame(tmp_path),
        output_path=tmp_path / "mask.png",
    )
    tight = receipt["object_pixel_bounds_xyxy"]
    admitted = receipt["pixel_bounds_xyxy"]
    inset = MAX_INNER_FEATHER_RADIUS_PIXELS
    assert admitted[0] + inset <= tight[0]
    assert admitted[1] + inset <= tight[1]
    assert admitted[2] - inset >= tight[2]
    assert admitted[3] - inset >= tight[3]


def test_margin_is_clamped_to_the_frame(tmp_path) -> None:
    """An object at the frame edge cannot push the support out of bounds."""

    receipt = _project_registered_bounds_mask(
        minimum_xyz=[-0.6, -0.6, 1.0],
        maximum_xyz=[0.6, 0.6, 1.2],
        camera=_camera(),
        frame_path=_frame(tmp_path),
        output_path=tmp_path / "mask.png",
    )
    left, top, right, bottom = receipt["pixel_bounds_xyxy"]
    assert left >= 0 and top >= 0
    assert right <= 256 and bottom <= 256
    assert right > left and bottom > top


def test_projection_provenance_is_unchanged(tmp_path) -> None:
    """Widening the support must not relabel where the geometry came from."""

    receipt = _project_registered_bounds_mask(
        minimum_xyz=[-0.05, -0.05, 1.0],
        maximum_xyz=[0.05, 0.05, 1.2],
        camera=_camera(),
        frame_path=_frame(tmp_path),
        output_path=tmp_path / "mask.png",
    )
    assert (
        receipt["projection_kind"]
        == "registered_world_aabb_conservative_projection"
    )
    assert receipt["observed_segmentation_truth"] is False
