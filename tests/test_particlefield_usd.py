"""Tests for blueprint_pipeline.particlefield_usd.

The convention math is pure-numpy and tested here; the USD writer needs pxr/usd-core
(skipped when absent — it's exercised via the usd-core venv / Isaac pod).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import SplatData
from blueprint_pipeline.particlefield_usd import (
    build_particlefield_arrays,
    write_particlefield_usd,
)

_HAS_PXR = importlib.util.find_spec("pxr") is not None


def _splat(n: int = 6, with_rest: bool = False) -> tuple[SplatData, np.ndarray | None]:
    rng = np.random.default_rng(3)
    splat = SplatData(
        count=n,
        xyz=rng.standard_normal((n, 3)).astype(np.float32),
        opacity=rng.standard_normal(n).astype(np.float32),
        f_dc=rng.standard_normal((n, 3)).astype(np.float32),
        scales=rng.standard_normal((n, 3)).astype(np.float32),
        quats=rng.standard_normal((n, 4)).astype(np.float32),
        properties=(),
    )
    rest = rng.standard_normal((n, 45)).astype(np.float32) if with_rest else None
    return splat, rest


def test_conventions_exp_sigmoid_normalized() -> None:
    splat, _ = _splat()
    arr = build_particlefield_arrays(splat)
    np.testing.assert_allclose(arr["scales"], np.exp(splat.scales), rtol=1e-5)
    np.testing.assert_allclose(arr["opacities"], 1.0 / (1.0 + np.exp(-splat.opacity)), rtol=1e-5)
    norms = np.linalg.norm(arr["orientations"], axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)
    assert np.all((arr["opacities"] >= 0) & (arr["opacities"] <= 1))


def test_degree0_dc_only() -> None:
    splat, _ = _splat(5)
    arr = build_particlefield_arrays(splat)
    assert arr["sh_degree"] == 0
    assert arr["sh_coefficients"].shape == (5, 3)
    np.testing.assert_allclose(arr["sh_coefficients"], splat.f_dc, rtol=1e-6)


def test_full_sh_layout_degree3() -> None:
    n = 4
    splat, _ = _splat(n)
    # construct f_rest so coeff k (1..15) per channel is identifiable: R=k, G=100+k, B=200+k
    rest = np.zeros((n, 45), dtype=np.float32)
    for k in range(15):
        rest[:, k] = k + 1            # R band
        rest[:, 15 + k] = 100 + k + 1  # G band
        rest[:, 30 + k] = 200 + k + 1  # B band
    arr = build_particlefield_arrays(splat, sh_rest=rest)
    assert arr["sh_degree"] == 3
    assert arr["sh_coefficients"].shape == (n * 16, 3)
    coeffs = arr["sh_coefficients"].reshape(n, 16, 3)
    np.testing.assert_allclose(coeffs[:, 0, :], splat.f_dc, rtol=1e-6)  # DC first
    # coeff 1 == (R_0, G_0, B_0) == (1, 101, 201)
    np.testing.assert_allclose(coeffs[0, 1, :], [1.0, 101.0, 201.0], rtol=1e-6)
    np.testing.assert_allclose(coeffs[0, 15, :], [15.0, 115.0, 215.0], rtol=1e-6)


def test_extent_is_aabb() -> None:
    splat, _ = _splat(20)
    arr = build_particlefield_arrays(splat)
    np.testing.assert_allclose(arr["extent"][0], splat.xyz.min(0), rtol=1e-6)
    np.testing.assert_allclose(arr["extent"][1], splat.xyz.max(0), rtol=1e-6)


def test_write_blocked_without_pxr_or_writes_usd(tmp_path: Path) -> None:
    splat, _ = _splat(8)
    out = tmp_path / "scene.usd"
    result = write_particlefield_usd(splat, out)
    if not _HAS_PXR:
        assert result["status"] == "blocked"
        assert "usd_core_unavailable" in result["blockers"]
    else:
        assert result["status"] == "completed"
        assert out.is_file() and result["splat_count"] == 8
        assert result["schema"] == "ParticleField3DGaussianSplat"
