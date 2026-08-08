from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.articulated_member_observation import (
    ArticulatedMemberObservationError,
    observe_horizontal_member_seam,
)


def _camera() -> dict:
    return {
        "pos": [0.0, 3.0, 1.0],
        "target": [0.0, 0.0, 1.0],
        "up": [0.0, 0.0, 1.0],
        "fov": 60.0,
    }


def test_observed_horizontal_seam_is_projected_into_world_geometry(tmp_path) -> None:
    pixels = np.full((150, 200, 3), 220, dtype=np.uint8)
    pixels[75:, :, :] = 80
    image = tmp_path / "front.png"
    Image.fromarray(pixels).save(image)

    observed = observe_horizontal_member_seam(
        image_path=image,
        camera_spec=_camera(),
        target_world_aabb_min_m=[-1.0, 0.8, 0.0],
        target_world_aabb_max_m=[1.0, 1.0, 2.0],
        front_plane_axis=1,
        front_plane_value_m=1.0,
    )

    assert observed["status"] == "observed_candidate_geometry"
    assert observed["seam_pixel"]["v"] == 74
    assert observed["seam_world_point_m"][1] == pytest.approx(1.0)
    assert observed["seam_world_point_m"][2] == pytest.approx(1.0077, abs=0.01)
    assert observed["edge_scan"]["peak_gradient_255"] > 100.0
    assert observed["receipt_digest"].startswith("sha256:")
    assert observed["claim_boundary"]["joint_topology_or_axis_proven"] is False


def test_low_contrast_scene_abstains_instead_of_inventing_seam(tmp_path) -> None:
    image = tmp_path / "flat.png"
    Image.fromarray(np.full((150, 200, 3), 120, dtype=np.uint8)).save(image)

    with pytest.raises(
        ArticulatedMemberObservationError, match="seam_edge_quality_insufficient"
    ):
        observe_horizontal_member_seam(
            image_path=image,
            camera_spec=_camera(),
            target_world_aabb_min_m=[-1.0, 0.8, 0.0],
            target_world_aabb_max_m=[1.0, 1.0, 2.0],
            front_plane_axis=1,
            front_plane_value_m=1.0,
        )
