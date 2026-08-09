from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.articulated_member_observation import (
    ArticulatedMemberObservationError,
    HANDLE_BAND_SCHEMA_VERSION,
    _camera_basis,
    _project,
    observe_front_plane_handle_band,
)


FRONT_Y = 1.829218141
CAMERA = {
    "pos": [1.9742142, 3.4, 1.15],
    "target": [1.9742142, FRONT_Y, 1.15],
    "up": [0.0, 0.0, 1.0],
    "fov": 55.0,
}
SEARCH_X = (1.617248144, 2.331180256)
SEARCH_Z = (0.939981249, 1.35)
HANDLE_X = (1.68, 1.90)
HANDLE_Z = (1.02, 1.08)


def _paint_fixture(path: Path, *, with_handle: bool) -> Path:
    width = height = 480
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :] = (232, 216, 210)
    if with_handle:
        position, forward, right, up, fov = _camera_basis(CAMERA)
        corners = [
            np.asarray([x, FRONT_Y, z])
            for x in HANDLE_X
            for z in HANDLE_Z
        ]
        projected = [
            _project(
                corner,
                position=position,
                forward=forward,
                right=right,
                up=up,
                vertical_fov_degrees=fov,
                width=width,
                height=height,
            )
            for corner in corners
        ]
        u_values = [point[0] for point in projected]
        v_values = [point[1] for point in projected]
        u_low, u_high = int(round(min(u_values))), int(round(max(u_values)))
        v_low, v_high = int(round(min(v_values))), int(round(max(v_values)))
        pixels[v_low : v_high + 1, u_low : u_high + 1] = (250, 250, 250)
    Image.fromarray(pixels).save(path)
    return path


def test_handle_band_observation_recovers_painted_world_band(tmp_path: Path) -> None:
    image = _paint_fixture(tmp_path / "front.png", with_handle=True)

    receipt = observe_front_plane_handle_band(
        image_path=image,
        camera_spec=CAMERA,
        front_plane_axis=1,
        front_plane_value_m=FRONT_Y,
        search_x_interval_m=list(SEARCH_X),
        search_z_interval_m=list(SEARCH_Z),
    )

    assert receipt["schema_version"] == HANDLE_BAND_SCHEMA_VERSION
    observed_z = receipt["observed_world_z_interval_m"]
    observed_x = receipt["observed_world_x_interval_m"]
    assert observed_z[0] == pytest.approx(HANDLE_Z[0], abs=0.02)
    assert observed_z[1] == pytest.approx(HANDLE_Z[1], abs=0.02)
    assert observed_x[0] == pytest.approx(HANDLE_X[0], abs=0.02)
    assert observed_x[1] == pytest.approx(HANDLE_X[1], abs=0.02)
    assert receipt["claim_boundary"]["protrusion_depth_observed"] is False
    assert receipt["claim_boundary"]["physical_site_metrology"] is False
    assert receipt["image_sha256"].startswith("sha256:")
    assert receipt["receipt_digest"].startswith("sha256:")


def test_handle_band_observation_fails_closed_without_band(tmp_path: Path) -> None:
    image = _paint_fixture(tmp_path / "empty.png", with_handle=False)

    with pytest.raises(ArticulatedMemberObservationError) as excinfo:
        observe_front_plane_handle_band(
            image_path=image,
            camera_spec=CAMERA,
            front_plane_axis=1,
            front_plane_value_m=FRONT_Y,
            search_x_interval_m=list(SEARCH_X),
            search_z_interval_m=list(SEARCH_Z),
        )

    assert any(
        "handle_band_not_observed" in error for error in excinfo.value.errors
    )
