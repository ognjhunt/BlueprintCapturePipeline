"""Hermetic tests for splat-occupancy carving of coarse label AABBs."""
from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import SplatData
from blueprint_pipeline.scene_placement import SceneObject
from blueprint_pipeline.splat_occupancy import (
    build_floor_occupancy_grid,
    refine_coarse_obstacles,
)


def _splat_from_points(points: np.ndarray) -> SplatData:
    n = points.shape[0]
    return SplatData(
        count=n,
        xyz=points.astype(np.float32),
        opacity=np.full(n, 5.0, dtype=np.float32),  # sigmoid ~= 0.993, solidly opaque
        f_dc=np.zeros((n, 3), dtype=np.float32),
        scales=np.zeros((n, 3), dtype=np.float32),
        quats=np.zeros((n, 4), dtype=np.float32),
        properties=(),
    )


def _obj(oid, label, bmin, bmax):
    centroid = tuple(0.5 * (a + b) for a, b in zip(bmin, bmax))
    return SceneObject(
        id=oid, label=label, bbox_min=bmin, bbox_max=bmax, centroid=centroid,
        source="interiorgs_labels",
    )


def _l_shaped_points() -> np.ndarray:
    """Dense mass along an L (two 0.5m-deep runs) inside a 3x3 jumbo box."""
    pts = []
    for x in np.arange(0.05, 3.0, 0.05):
        for y in np.arange(0.05, 0.5, 0.05):
            for z in (0.3, 0.6, 0.9):
                pts.append((x, y, z))
    for y in np.arange(0.5, 3.0, 0.05):
        for x in np.arange(0.05, 0.5, 0.05):
            for z in (0.3, 0.6, 0.9):
                pts.append((x, y, z))
    return np.array(pts, dtype=np.float32)


class TestOccupancyGrid:
    def test_floor_and_ceiling_splats_excluded(self):
        pts = np.array([
            [1.0, 1.0, 0.01],   # floor splat: below the band
            [1.0, 1.0, 2.0],    # ceiling fixture: above the band
            [1.0, 1.0, 0.6],    # body-height mass
        ])
        grid = build_floor_occupancy_grid(_splat_from_points(pts), floor_z=0.0)
        assert float(grid.weights.sum()) == pytest.approx(
            1.0 / (1.0 + np.exp(-5.0)), rel=1e-3
        )

    def test_empty_band_yields_empty_grid(self):
        pts = np.array([[0.0, 0.0, 5.0]])
        grid = build_floor_occupancy_grid(_splat_from_points(pts), floor_z=0.0)
        assert grid.weights.sum() == 0.0


class TestRefineCoarseObstacles:
    def test_l_shape_carves_open_corner(self):
        splat = _splat_from_points(_l_shaped_points())
        jumbo = _obj("105", "cupboard", (0.0, 0.0, 0.0), (3.0, 3.0, 1.2))
        refined, report = refine_coarse_obstacles([jumbo], splat, floor_z=0.0)
        assert report["refined"][0]["kept_coarse_box"] is False
        assert len(refined) > 1
        # The open corner of the L (around x=2, y=2) must have NO obstacle column.
        assert not any(
            o.bbox_min[0] < 2.2 and o.bbox_max[0] > 1.8
            and o.bbox_min[1] < 2.2 and o.bbox_max[1] > 1.8
            for o in refined
        )
        # The dense runs are still covered.
        assert any(o.bbox_min[0] < 1.5 < o.bbox_max[0] and o.bbox_min[1] < 0.3 for o in refined)
        # Column z extents follow the splat mass, not the full label box.
        tops = {round(o.bbox_max[2], 2) for o in refined}
        assert tops == {0.9}

    def test_small_boxes_pass_through(self):
        splat = _splat_from_points(_l_shaped_points())
        small = _obj("88", "pot", (1.0, 1.0, 0.8), (1.2, 1.2, 0.9))
        refined, _ = refine_coarse_obstacles([small], splat, floor_z=0.0)
        assert refined == [small]

    def test_walls_never_refined(self):
        splat = _splat_from_points(_l_shaped_points())
        wall = _obj("wall_1", "wall", (0.0, 0.0, 0.0), (6.0, 0.24, 2.6))
        refined, _ = refine_coarse_obstacles([wall], splat, floor_z=0.0)
        assert refined == [wall]

    def test_jumbo_without_splat_mass_keeps_coarse_box(self):
        splat = _splat_from_points(_l_shaped_points())
        far_jumbo = _obj("200", "bed", (50.0, 50.0, 0.0), (53.0, 53.0, 0.6))
        refined, report = refine_coarse_obstacles([far_jumbo], splat, floor_z=0.0)
        assert refined == [far_jumbo]
        assert report["refined"][0]["kept_coarse_box"] is True
