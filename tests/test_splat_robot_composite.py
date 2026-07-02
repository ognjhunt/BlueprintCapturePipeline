"""Hermetic tests for the local robot-into-splat depth compositor."""
from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.gaussian_splat_decode import SplatData
from blueprint_pipeline.splat_robot_composite import (
    _range_to_z_depth,
    composite_robot_into_splat,
)


def _cam(width=64, height=48, vfov_deg=60.0):
    return {
        "eye": (0.0, 0.0, 1.0),
        "target": (5.0, 0.0, 1.0),
        "up": (0.0, 0.0, 1.0),
        "vfov": math.radians(vfov_deg),
        "width": width,
        "height": height,
    }


def _splat_wall(x_plane: float) -> SplatData:
    pts = np.array(
        [[x_plane, y, z] for y in np.arange(-2, 2, 0.05) for z in np.arange(0.0, 2.0, 0.05)],
        dtype=np.float32,
    )
    n = pts.shape[0]
    return SplatData(
        count=n, xyz=pts, opacity=np.full(n, 5.0, np.float32),
        f_dc=np.zeros((n, 3), np.float32), scales=np.zeros((n, 3), np.float32),
        quats=np.zeros((n, 4), np.float32), properties=(),
    )


class TestRangeToZ:
    def test_on_axis_range_equals_z(self):
        d = np.full((48, 64), 3.0)
        z = _range_to_z_depth(d, math.radians(60), 64, 48)
        assert z[24, 32] == pytest.approx(3.0, abs=0.01)
        # Off-axis pixels have z < range.
        assert z[0, 0] < 3.0


class TestComposite:
    def _files(self, tmp_path, robot_range: float):
        cam = _cam()
        scene = np.full((48, 64, 3), 200, dtype=np.uint8)
        robot = np.zeros((48, 64, 3), dtype=np.uint8)
        robot[20:30, 28:36] = (255, 0, 0)  # a red robot patch
        distance = np.full((48, 64), np.inf)
        distance[20:30, 28:36] = robot_range
        scene_p = tmp_path / "scene.png"; Image.fromarray(scene).save(scene_p)
        robot_p = tmp_path / "robot.png"; Image.fromarray(robot).save(robot_p)
        dist_p = tmp_path / "dist.npy"; np.save(dist_p, distance)
        return cam, scene_p, robot_p, dist_p

    def test_robot_in_front_of_wall_visible(self, tmp_path):
        cam, scene_p, robot_p, dist_p = self._files(tmp_path, robot_range=2.0)
        report = composite_robot_into_splat(
            scene_p, robot_p, dist_p, cam, _splat_wall(4.0), tmp_path / "out.png",
        )
        assert report["visible_robot_pixels"] == report["robot_pixels"] > 0
        out = np.asarray(Image.open(tmp_path / "out.png"))
        assert (out[25, 32] == (255, 0, 0)).all()

    def test_robot_behind_wall_occluded(self, tmp_path):
        cam, scene_p, robot_p, dist_p = self._files(tmp_path, robot_range=6.0)
        report = composite_robot_into_splat(
            scene_p, robot_p, dist_p, cam, _splat_wall(4.0), tmp_path / "out.png",
        )
        assert report["visible_robot_pixels"] == 0
        assert report["occluded_robot_pixels"] == report["robot_pixels"] > 0
        out = np.asarray(Image.open(tmp_path / "out.png"))
        assert (out[25, 32] == (200, 200, 200)).all()

    def test_splat_hole_does_not_occlude(self, tmp_path):
        # No splat mass on these rays at all: robot must win.
        cam, scene_p, robot_p, dist_p = self._files(tmp_path, robot_range=6.0)
        empty = _splat_wall(4.0)
        far_left = SplatData(
            count=1, xyz=np.array([[4.0, -50.0, 1.0]], np.float32),
            opacity=np.full(1, 5.0, np.float32), f_dc=np.zeros((1, 3), np.float32),
            scales=np.zeros((1, 3), np.float32), quats=np.zeros((1, 4), np.float32),
            properties=(),
        )
        report = composite_robot_into_splat(
            scene_p, robot_p, dist_p, cam, far_left, tmp_path / "out.png",
        )
        assert report["visible_robot_pixels"] == report["robot_pixels"] > 0
