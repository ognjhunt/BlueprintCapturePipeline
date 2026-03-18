"""Tests for plucker_rays.py — 6-channel Plücker ray map generation."""
from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.synthesis.plucker_rays import (
    compute_plucker_map,
    normalise_plucker,
)


@pytest.fixture
def simple_intrinsics():
    return {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}


class TestComputePluckerMap:
    def test_output_shape(self, simple_intrinsics):
        T = np.eye(4, dtype=np.float64)
        out = compute_plucker_map(T_world_camera=T, intrinsics=simple_intrinsics, height=48, width=64)
        assert out.shape == (6, 48, 64)
        assert out.dtype == np.float32

    def test_direction_channels_are_unit_vectors(self, simple_intrinsics):
        """Channels [0:3] must be unit direction vectors (L2 norm = 1)."""
        T = np.eye(4, dtype=np.float64)
        out = compute_plucker_map(T_world_camera=T, intrinsics=simple_intrinsics, height=48, width=64)
        d = out[:3]  # [3, H, W]
        norms = np.linalg.norm(d, axis=0)  # [H, W]
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_moments_zero_at_origin(self, simple_intrinsics):
        """Camera at world origin → moment m = O × d = 0 × d = 0 for all pixels."""
        T = np.eye(4, dtype=np.float64)  # translation = (0,0,0)
        out = compute_plucker_map(T_world_camera=T, intrinsics=simple_intrinsics, height=32, width=32)
        m = out[3:]  # [3, H, W]
        np.testing.assert_allclose(m, 0.0, atol=1e-5)

    def test_nonzero_translation_produces_nonzero_moments(self, simple_intrinsics):
        """Camera at non-origin position → moments must not all be zero."""
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = 3.0   # 3 m offset in X
        out = compute_plucker_map(T_world_camera=T, intrinsics=simple_intrinsics, height=32, width=32)
        m = out[3:]
        assert not np.allclose(m, 0.0, atol=1e-5)


class TestNormalisePlucker:
    def test_output_shape_unchanged(self, simple_intrinsics):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = 2.0
        raw = compute_plucker_map(T_world_camera=T, intrinsics=simple_intrinsics, height=32, width=32)
        normalised = normalise_plucker(raw)
        assert normalised.shape == raw.shape
        assert normalised.dtype == np.float32

    def test_direction_channels_unchanged(self, simple_intrinsics):
        """Normalisation must not alter the direction channels (already unit)."""
        T = np.eye(4, dtype=np.float64)
        T[1, 3] = 1.5
        raw = compute_plucker_map(T_world_camera=T, intrinsics=simple_intrinsics, height=32, width=32)
        normalised = normalise_plucker(raw)
        np.testing.assert_allclose(normalised[:3], raw[:3], atol=1e-6)
