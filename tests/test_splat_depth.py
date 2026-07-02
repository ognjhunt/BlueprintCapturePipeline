"""Hermetic tests for the point-splat depth renderer (no DA3, no GPU)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import SplatData
from blueprint_pipeline.scene_placement.perception_index import (
    PerceptionSceneSpatialIndex,
)
from blueprint_pipeline.splat_depth import (
    depth_provider_for_camera,
    render_pointsplat_depth,
)


def _splat(points, opacity_logit=5.0):
    pts = np.asarray(points, dtype=np.float32)
    n = pts.shape[0]
    return SplatData(
        count=n,
        xyz=pts,
        opacity=np.full(n, opacity_logit, dtype=np.float32),
        f_dc=np.zeros((n, 3), dtype=np.float32),
        scales=np.zeros((n, 3), dtype=np.float32),
        quats=np.zeros((n, 4), dtype=np.float32),
        properties=(),
    )


def _camera(eye, target, *, width=320, height=240, vfov_deg=60.0):
    return {
        "eye": eye,
        "target": target,
        "up": (0.0, 0.0, 1.0),
        "vfov": math.radians(vfov_deg),
        "width": width,
        "height": height,
    }


class TestRenderPointsplatDepth:
    def test_on_axis_point_has_exact_z_depth(self):
        cam = _camera((0.0, 0.0, 1.0), (5.0, 0.0, 1.0))
        splat = _splat([[3.0, 0.0, 1.0]])  # 3m straight ahead
        depth = render_pointsplat_depth(splat, cam, depth_scale=4)
        h, w = depth.shape
        assert depth[h // 2, w // 2] == pytest.approx(3.0, abs=1e-6)

    def test_z_depth_not_range_for_off_axis_point(self):
        cam = _camera((0.0, 0.0, 1.0), (5.0, 0.0, 1.0))
        # 3m forward, 1m to the side: z-depth is 3.0, Euclidean range sqrt(10).
        splat = _splat([[3.0, 1.0, 1.0]])
        depth = render_pointsplat_depth(splat, cam, depth_scale=1)
        finite = depth[np.isfinite(depth)]
        # Hole-fill dilates the lone point to its neighborhood; every filled
        # value must still be the z-depth 3.0, never the Euclidean range.
        assert finite.size >= 1
        assert np.allclose(finite, 3.0, atol=1e-6)

    def test_nearest_point_wins_the_pixel(self):
        cam = _camera((0.0, 0.0, 1.0), (5.0, 0.0, 1.0))
        splat = _splat([[2.0, 0.0, 1.0], [4.0, 0.0, 1.0]])  # same ray, two depths
        depth = render_pointsplat_depth(splat, cam, depth_scale=4)
        h, w = depth.shape
        assert depth[h // 2, w // 2] == pytest.approx(2.0, abs=1e-6)

    def test_transparent_and_behind_points_ignored(self):
        cam = _camera((0.0, 0.0, 1.0), (5.0, 0.0, 1.0))
        behind = _splat([[-2.0, 0.0, 1.0]])
        assert not np.isfinite(render_pointsplat_depth(behind, cam)).any()
        transparent = _splat([[3.0, 0.0, 1.0]], opacity_logit=-5.0)  # sigmoid ~0.007
        assert not np.isfinite(render_pointsplat_depth(transparent, cam)).any()

    def test_hole_fill_median(self):
        cam = _camera((0.0, 0.0, 1.0), (5.0, 0.0, 1.0), width=64, height=64)
        # A small cluster: fills its own pixel plus neighbors via median passes.
        splat = _splat([[3.0, dx, 1.0 + dz] for dx in (-0.05, 0.0, 0.05) for dz in (-0.05, 0.0, 0.05)])
        depth = render_pointsplat_depth(splat, cam, depth_scale=4)
        assert np.isfinite(depth).sum() >= 9


class TestEndToEndUnprojection:
    def test_detection_unprojects_back_to_world_position(self):
        """A wall of splats at x=3 + a 2D box over it must yield an AABB near x=3."""
        cam = _camera((0.0, 0.0, 1.0), (5.0, 0.0, 1.0), width=320, height=240)
        wall_pts = [
            [3.0, y, z]
            for y in np.arange(-1.0, 1.0, 0.05)
            for z in np.arange(0.4, 1.6, 0.05)
        ]
        splat = _splat(wall_pts)
        provider = depth_provider_for_camera(splat, cam, depth_scale=4)
        # A detection box in the middle of the image (the wall fills the frame).
        det = {"label": "wall_thing", "bbox_px": (140, 100, 180, 140), "confidence": 0.9}
        index = PerceptionSceneSpatialIndex([det], provider, cam)
        objects = index.objects()
        assert len(objects) == 1
        obj = objects[0]
        assert obj.centroid[0] == pytest.approx(3.0, abs=0.15)
        assert abs(obj.centroid[1]) < 0.5
