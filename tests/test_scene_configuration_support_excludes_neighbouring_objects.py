"""The removal support may hold the contact shadow, never the neighbour.

Widening the admitted support to the object's full visual footprint fixed a
residual contact shadow, but the widened region is a plain rectangle: in scene
839873 it reached across a laptop standing beside the mug, and the independent
reviewer rejected the frame because the edit smeared the laptop.

A contact shadow is a darkening of the surface the object rests on, and it
touches the object that casts it.  A neighbouring object is neither.  These
tests pin that distinction, and pin that the object's own box is never subject
to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import (
    SUPPORTING_SURFACE_SHADOW_LUMINANCE_FLOOR_FRACTION,
    _constrain_margin_to_supporting_surface,
)


SURFACE = 240
SHADOW = 200
NEIGHBOUR = 30

WIDENED = (10, 10, 90, 90)
OBJECT = (30, 30, 70, 70)


def _frame(tmp_path: Path, *, paint: dict[tuple[int, int, int, int], int]) -> Path:
    """Render a synthetic scene: a bright surface with regions painted on it."""

    canvas = np.full((100, 100), SURFACE, dtype=np.uint8)
    for (left, top, right, bottom), value in paint.items():
        canvas[top:bottom, left:right] = value
    path = tmp_path / "frame.png"
    Image.fromarray(canvas, mode="L").save(path)
    return path


def _support(frame_path: Path) -> tuple[np.ndarray, dict]:
    mask = np.zeros((100, 100), dtype=np.uint8)
    left, top, right, bottom = WIDENED
    mask[top:bottom, left:right] = 255
    receipt = _constrain_margin_to_supporting_surface(
        mask=mask,
        frame_path=frame_path,
        widened_bounds=WIDENED,
        object_bounds=OBJECT,
    )
    return mask, receipt


def test_contact_shadow_stays_inside_the_admitted_support(tmp_path: Path) -> None:
    """A soft darkening of the surface touching the object is still admitted."""

    frame = _frame(tmp_path, paint={(20, 30, 30, 70): SHADOW})

    mask, receipt = _support(frame)

    assert (mask[30:70, 20:30] > 0).all()
    assert receipt["margin_pixels_admitted"] == receipt["margin_pixels_offered"]
    assert receipt["rule"] == "supporting_surface_luminance_band_connected_to_object"


def test_neighbouring_object_is_dropped_from_the_support(tmp_path: Path) -> None:
    """A dark object standing in the margin is not the object's soft boundary."""

    frame = _frame(tmp_path, paint={(72, 10, 90, 90): NEIGHBOUR})

    # The widened rectangle -- what the support was before this constraint --
    # covers the neighbour outright. That is the defect being fixed, so state
    # it here rather than trusting that the rectangle happened to miss.
    rectangle = np.zeros((100, 100), dtype=np.uint8)
    left, top, right, bottom = WIDENED
    rectangle[top:bottom, left:right] = 255
    assert (rectangle[10:90, 72:90] > 0).any()

    mask, receipt = _support(frame)

    assert not (mask[10:90, 72:90] > 0).any()
    assert receipt["margin_pixels_admitted"] < receipt["margin_pixels_offered"]
    assert receipt["surface_reference_luminance"] == pytest.approx(SURFACE, abs=1.0)


def test_the_objects_own_box_is_always_admitted(tmp_path: Path) -> None:
    """The object may be darker than its surface; it is still what we remove.

    Applying the surface test to the object itself would refuse to remove any
    dark object, which is the opposite of the edit's purpose.
    """

    frame = _frame(tmp_path, paint={OBJECT: NEIGHBOUR})

    mask, _receipt = _support(frame)

    left, top, right, bottom = OBJECT
    assert (mask[top:bottom, left:right] > 0).all()


def test_detached_surface_coloured_region_is_not_admitted(tmp_path: Path) -> None:
    """Tone alone is not enough; a shadow touches the object that casts it.

    A bright wall seen past the object is tone-compatible with the tabletop.
    Only growing outward from the object keeps it out of the support.
    """

    frame = _frame(
        tmp_path,
        paint={(10, 10, 90, 22): NEIGHBOUR, (10, 10, 90, 14): SURFACE},
    )

    mask, _receipt = _support(frame)

    assert not (mask[10:14, 10:90] > 0).any()


def test_support_never_grows_beyond_the_widened_rectangle(tmp_path: Path) -> None:
    """The seal proves nothing outside the digest-bound support changed.

    Narrowing the support keeps that proof valid; widening it would silently
    void the guarantee the seal reports.
    """

    frame = _frame(tmp_path, paint={})

    mask, _receipt = _support(frame)

    outside = mask.copy()
    left, top, right, bottom = WIDENED
    outside[top:bottom, left:right] = 0
    assert not (outside > 0).any()


def test_floor_is_stated_relative_to_the_measured_surface(tmp_path: Path) -> None:
    """The band is reported so a rejected frame can be argued about."""

    frame = _frame(tmp_path, paint={(72, 10, 90, 90): NEIGHBOUR})

    _mask, receipt = _support(frame)

    assert receipt["luminance_floor"] == pytest.approx(
        receipt["surface_reference_luminance"]
        * SUPPORTING_SURFACE_SHADOW_LUMINANCE_FLOOR_FRACTION,
        rel=1e-6,
    )
    assert NEIGHBOUR < receipt["luminance_floor"] < SHADOW
