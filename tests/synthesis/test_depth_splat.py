"""Tests for depth_splat.py — forward splatting reference → target viewpoint."""
from __future__ import annotations

import io

import numpy as np
import pytest

from blueprint_pipeline.synthesis.depth_splat import depth_splat, load_depth_png

# Use constants from conftest
DEPTH_H, DEPTH_W = 192, 256
IMAGE_H, IMAGE_W = 1440, 1920


class TestLoadDepthPng:
    def test_uint16_loads_as_metres(self, synthetic_depth_png_bytes, tmp_path):
        """uint16 PNG values are millimetres; loading with scale=0.001 → metres."""
        p = tmp_path / "depth.png"
        p.write_bytes(synthetic_depth_png_bytes)
        depth = load_depth_png(p, depth_scale=0.001)
        assert depth.dtype == np.float32
        assert depth.shape == (DEPTH_H, DEPTH_W)
        # Values should be in roughly [0.5, 5.0] metres (500–5000 mm fixture)
        valid = depth[depth > 0]
        assert valid.min() >= 0.49
        assert valid.max() <= 5.01

    def test_zero_pixels_remain_zero(self, synthetic_depth_png_bytes, tmp_path):
        """Pixel (0,0) was set to 0 in the fixture; must stay 0 after loading."""
        p = tmp_path / "depth.png"
        p.write_bytes(synthetic_depth_png_bytes)
        depth = load_depth_png(p, depth_scale=0.001)
        assert depth[0, 0] == 0.0


class TestDepthSplat:
    def _make_small_scene(self, h=64, w=64):
        """Return a simple flat-plane scene: all pixels at 2.0 m depth."""
        rng = np.random.default_rng(99)
        image = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        depth = np.full((h, w), 2.0, dtype=np.float32)
        intrinsics = {"fx": 60.0, "fy": 60.0, "cx": float(w) / 2, "cy": float(h) / 2}
        return image, depth, intrinsics

    def test_identity_warp_is_identity(self):
        """When ref and target share the same pose, splatted image ≈ ref image."""
        image, depth, K = self._make_small_scene()
        T = np.eye(4, dtype=np.float64)
        warped, mask = depth_splat(
            ref_image=image,
            ref_depth=depth,
            T_world_ref=T,
            K_ref=K,
            T_world_target=T,
            K_target=K,
            target_h=image.shape[0],
            target_w=image.shape[1],
            fill_holes=False,
        )
        assert warped.shape == image.shape
        assert mask.dtype == bool
        # At identity, every valid-depth pixel should land on itself
        assert mask.sum() > 0
        # Central crop should match exactly (boundary pixels may miss due to rounding)
        cy, cx = image.shape[0] // 2, image.shape[1] // 2
        s = 10
        np.testing.assert_array_equal(
            warped[cy - s : cy + s, cx - s : cx + s],
            image[cy - s : cy + s, cx - s : cx + s],
        )

    def test_zero_depth_pixels_excluded(self):
        """Pixels with depth=0 (invalid) must not be projected."""
        image, depth, K = self._make_small_scene()
        depth[:] = 0.0  # all invalid
        T = np.eye(4, dtype=np.float64)
        warped, mask = depth_splat(
            ref_image=image,
            ref_depth=depth,
            T_world_ref=T,
            K_ref=K,
            T_world_target=T,
            K_target=K,
            target_h=image.shape[0],
            target_w=image.shape[1],
            fill_holes=False,
        )
        # No valid pixels → output is black, mask is all False
        assert not mask.any()
        assert warped.sum() == 0

    def test_output_shape_matches_target(self):
        """Output shape must be (target_h, target_w, 3) regardless of ref resolution."""
        image, depth, K = self._make_small_scene(h=32, w=48)
        T = np.eye(4, dtype=np.float64)
        target_h, target_w = 16, 24
        K_target = {"fx": 30.0, "fy": 30.0, "cx": 12.0, "cy": 8.0}
        warped, mask = depth_splat(
            ref_image=image,
            ref_depth=depth,
            T_world_ref=T,
            K_ref=K,
            T_world_target=T,
            K_target=K_target,
            target_h=target_h,
            target_w=target_w,
            fill_holes=False,
        )
        assert warped.shape == (target_h, target_w, 3)
        assert mask.shape == (target_h, target_w)

    def test_hole_filling_runs_without_error(self):
        """Hole filling should not raise even if only half the pixels are covered."""
        image, depth, K = self._make_small_scene(h=32, w=32)
        # Zero out left half → will produce holes in the output
        depth[:, : depth.shape[1] // 2] = 0.0
        T = np.eye(4, dtype=np.float64)
        warped, mask = depth_splat(
            ref_image=image,
            ref_depth=depth,
            T_world_ref=T,
            K_ref=K,
            T_world_target=T,
            K_target=K,
            target_h=32,
            target_w=32,
            fill_holes=True,
            fill_radius=2,
        )
        assert warped.shape == (32, 32, 3)
        assert mask.shape == (32, 32)
        # After hole filling, more pixels should be covered than raw splat
        # (or at least it should not crash)
        assert not np.any(np.isnan(warped.astype(float)))
