from __future__ import annotations

import numpy as np

from blueprint_pipeline.external_scene_robot_placement import _infer_horizontal_support_surface


def test_dominant_horizontal_floor_rejects_low_geometry_outlier() -> None:
    vertices = np.asarray(
        [
            [-3.0, -3.0, 0.0],
            [3.0, -3.0, 0.0],
            [3.0, 3.0, 0.0],
            [-3.0, 3.0, 0.0],
            [0.0, 0.0, -2.0],
            [0.1, 0.0, -2.0],
            [0.0, 0.1, -2.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)

    floor_z, support_triangles, evidence = _infer_horizontal_support_surface(
        stage_vertices=vertices,
        faces=faces,
    )

    assert floor_z == 0.0
    assert support_triangles.shape[0] == 2
    assert evidence["global_minimum_vertex_z_rejected_as_floor"] == -2.0
    assert evidence["selection_method"] == "dominant_lower_horizontal_triangle_area_band"
